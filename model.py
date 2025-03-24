from typing import Optional, Union, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen2Model, Qwen2ForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import KwargsForCausalLM
from transformers.processing_utils import Unpack


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.kaiming_uniform_(self.a.data, a=self.alpha, mode='fan_in', nonlinearity='leaky_relu')
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layer_norm = torch.nn.LayerNorm(out_features, 1e-5, elementwise_affine=True)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # b,N, N_out_features
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # B, N , N
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)  # B, N, N
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        h_prime = self.layer_norm(h_prime)

        if self.concat:
            return F.gelu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[1]  # number of nodes
        B = Wh.size()[0]  # number of batches
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = Wh.repeat(1, N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=2)
        return all_combinations_matrix.view(B, N, N, 2 * self.out_features)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.2, alpha=0.02, nheads=2):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList(
            [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
        self.fc = nn.Linear(nhid, nhid)
        self.layer_norm = torch.nn.LayerNorm(nhid, 1e-5, elementwise_affine=True)

    def forward(self, x, adj):
        res = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.gelu(self.out_att(x, adj))
        x = self.fc(x)
        x = x + res
        x = self.layer_norm(x)
        return x


class Qwen2GoTModel(Qwen2Model):
    def __init__(self, config):
        super(Qwen2GoTModel, self).__init__(config)
        self.gat = GAT(nfeat=config.hidden_size, nhid=config.hidden_size)
        self.corss_attn = nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size,
                                                vdim=config.hidden_size, num_heads=1, batch_first=True)
        self.gate = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # Set weights to small values
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        # Set bias to a large negative value to ensure sigmoid outputs ~0 initially
        nn.init.constant_(self.gate.bias, -10.0)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            got_nodes: Optional[torch.FloatTensor] = None,
            adj_matrix: Optional[torch.IntTensor] = None,
            **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        # 获取原始嵌入
        word_embeds = self.embed_tokens(input_ids)  # [batch, seq, dim]

        if got_nodes is not None and adj_matrix is not None:
            # 通过图注意力网络处理节点嵌入
            node_embeds = self.gat(got_nodes, adj_matrix)  # [batch, node_num, dim]
            # 交叉注意力
            got_att, _ = self.corss_attn(
                word_embeds,
                node_embeds,
                node_embeds,
                key_padding_mask=(got_nodes.sum(dim=-1) == 0),
            )
            # 门控融合机制
            gate = torch.sigmoid(self.gate(
                torch.cat([word_embeds, got_att], dim=-1)
            ))
            # 融合嵌入
            fused_embeds = (1 - gate) * word_embeds + gate * got_att
            # 继续使用融合后的嵌入进行模型的后续计算
            outputs = super().forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=fused_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **flash_attn_kwargs,
            )
        else:
            # 如果没有图数据，则使用标准的输入
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **flash_attn_kwargs,
            )

        return outputs


class Qwen2GoTForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super(Qwen2GoTForCausalLM, self).__init__(config)
        # 替换原有模型为Qwen2GoTModel
        self.model = Qwen2GoTModel(config)
        # 确保权重正确初始化
        # self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            got_nodes: Optional[torch.FloatTensor] = None,
            adj_matrix: Optional[torch.IntTensor] = None,
            **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            got_nodes=got_nodes,
            adj_matrix=adj_matrix,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
