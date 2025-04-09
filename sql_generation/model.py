from typing import Optional, Union, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import Qwen2Model, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import is_torchdynamo_compiling, logging

logger = logging.get_logger(__name__)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layer_norm = torch.nn.LayerNorm(out_features, 1e-5, elementwise_affine=True)

    def post_init(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.kaiming_uniform_(self.a.data, a=self.alpha, mode='fan_in', nonlinearity='leaky_relu')

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

    def post_init(self):
        for attention in self.attentions:
            attention.post_init()
        self.out_att.post_init()

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


class GraphFused(nn.Module):
    def __init__(self, config):
        super(GraphFused, self).__init__()
        self.gat = GAT(nfeat=config.hidden_size, nhid=config.hidden_size)
        self.cross_attn = nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size,
                                                vdim=config.hidden_size, num_heads=8, batch_first=True)
        self.gate = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # Set weights to small values, and set bias to a large negative value to ensure sigmoid outputs ~0 initially
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.gate.bias, -10.0)

    def post_init(self):
        self.gat.post_init()

    def forward(
            self,
            word_embeds: torch.FloatTensor = None,
            node_embeds: torch.FloatTensor = None,
            got_nodes: torch.LongTensor = None,
            adj_matrix: torch.IntTensor = None,
    ):
        # 获取图嵌入
        graph_embeds = self.gat(node_embeds, adj_matrix)  # [batch, num_nodes, dim]
        # 生成 key_padding_mask，标记填充节点
        key_padding_mask = (got_nodes.sum(dim=-1) == 0)  # [batch, num_nodes]
        # 交叉注意力
        got_att, _ = self.cross_attn(
            word_embeds,
            graph_embeds,
            graph_embeds,
            key_padding_mask=key_padding_mask,
        )
        # 门控融合机制
        gate = torch.sigmoid(self.gate(
            torch.cat([word_embeds, got_att], dim=-1)
        ))
        # 融合嵌入
        return (1 - gate) * word_embeds + gate * got_att


class GoTSQLModel(Qwen2Model):
    def __init__(self, config):
        super(GoTSQLModel, self).__init__(config)
        self.graph_fused = GraphFused(config)

    def init_extracted_modules(self):
        self.graph_fused.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            got_nodes: Optional[torch.LongTensor] = None,
            adj_matrix: Optional[torch.IntTensor] = None,
    ):
        if got_nodes is not None and adj_matrix is not None:
            # 获取原始词嵌入
            word_embeds = self.embed_tokens(input_ids)  # [batch, seq, dim]
            # 获取结点嵌入
            node_embeds = self.get_node_embeds(got_nodes=got_nodes)
            # 融合嵌入
            fused_embeds = self.graph_fused(
                word_embeds=word_embeds,
                node_embeds=node_embeds,
                got_nodes=got_nodes,
                adj_matrix=adj_matrix
            )
            # 继续使用融合后的嵌入进行模型的后续计算
            return super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=fused_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
            )
        # 如果没有图数据，则使用标准的输入
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

    def get_node_embeds(self, got_nodes):
        batch_size, num_nodes, seq_len = got_nodes.size()
        flat_nodes = got_nodes.view(-1, seq_len)  # [batch*num_nodes, seq_len]
        flat_node_embeds = self.embed_tokens(flat_nodes)  # [batch*num_nodes, seq_len, dim]

        mask = (flat_nodes != 0).float()  # [batch*num_nodes, seq_len]
        expanded_mask = mask.unsqueeze(-1)  # [batch*num_nodes, seq_len, 1]
        # 对嵌入应用掩码(将填充位置的嵌入置为0)
        masked_embeds = flat_node_embeds * expanded_mask  # [batch*num_nodes, seq_len, dim]
        seq_lengths = torch.clamp(mask.sum(dim=1, keepdim=True).unsqueeze(-1), min=1.0)  # [batch*num_nodes, 1, 1]
        node_embeds = masked_embeds.sum(dim=1) / seq_lengths.squeeze(-1)  # [batch*num_nodes, dim]
        return node_embeds.view(batch_size, num_nodes, -1)  # [batch, num_nodes, dim]


class GoTSQLModelForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super(GoTSQLModelForCausalLM, self).__init__(config)
        # 替换原有模型为GoTSQLModel
        self.model = GoTSQLModel(config)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: int = 0,
            got_nodes: Optional[torch.LongTensor] = None,
            adj_matrix: Optional[torch.IntTensor] = None,
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
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            got_nodes=got_nodes,
            adj_matrix=adj_matrix,
        )

        hidden_states = outputs[0]
        if labels is None and not is_torchdynamo_compiling():
            logger.warning_once(
                "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
            )
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # TODO: remove the float() operation in v4.46
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

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
