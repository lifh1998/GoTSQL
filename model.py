from typing import Optional, Union, Tuple, List

from torch import nn
from transformers import Qwen2Model, Qwen2ForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import KwargsForCausalLM
from transformers.processing_utils import Unpack


class GraphAttentionLayer(nn.Module):
    # GAT层代码保持不变
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
    # GAT模型代码保持不变
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


class QwenGoTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.gat = GAT(nfeat=config.hidden_size, nhid=config.hidden_size)  # 复用原GAT模块
        self.gate = nn.Linear(2 * config.hidden_size, config.hidden_size)

    def forward(self, input_ids, got_nodes, adj_matrix):
        # 原始词嵌入
        word_embeds = self.word_embeddings(input_ids)  # [batch, seq, dim]
        # 图结构编码
        node_embeds = self.gat(got_nodes, adj_matrix)  # [batch, node_num, dim]
        # **门控融合机制**
        gate = torch.sigmoid(self.gate(
            torch.cat([word_embeds.mean(1), node_embeds.mean(1)], dim=-1)
        ))
        fused_embeds = (1 - gate) * word_embeds + gate * node_embeds.unsqueeze(1)
        return fused_embeds


import torch
import torch.nn.functional as F
from torch import nn


class GraphAttentionLayer(nn.Module):
    # GAT层代码保持不变
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
    # GAT模型代码保持不变
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


class QwenGoTEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.gat = GAT(nfeat=config.hidden_size, nhid=config.hidden_size)  # 复用原GAT模块
        self.gate = nn.Linear(2 * config.hidden_size, config.hidden_size)

    def forward(self, input_ids, got_nodes, adj_matrix):
        # 原始词嵌入
        word_embeds = self.word_embeddings(input_ids)  # [batch, seq, dim]
        # 图结构编码
        node_embeds = self.gat(got_nodes, adj_matrix)  # [batch, node_num, dim]
        # **门控融合机制**
        gate = torch.sigmoid(self.gate(
            torch.cat([word_embeds.mean(1), node_embeds.mean(1)], dim=-1)
        ))
        fused_embeds = (1 - gate) * word_embeds + gate * node_embeds.unsqueeze(1)
        return fused_embeds


class QwenGoTEnhancedEmbeddings(nn.Module):
    def __init__(self, config, original_embeddings):
        super().__init__()
        self.word_embeddings = original_embeddings
        self.gat = GAT(nfeat=config.hidden_size, nhid=config.hidden_size)
        # 确保使用正确的注意力头数
        num_heads = config.num_key_value_heads if hasattr(config, 'num_key_value_heads') else 8
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.gate_linear = nn.Linear(3 * config.hidden_size, config.hidden_size)
        self.output_linear = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.dtype = torch.float16  # 设置默认数据类型为半精度

    def forward(self, input_ids=None, got_nodes=None, adj_matrix=None):
        # 原始词嵌入
        word_embeds = self.word_embeddings(input_ids)
        # 如果没有提供图结构数据，则直接返回原始词嵌入
        if got_nodes is None or adj_matrix is None:
            return word_embeds
        try:
            # 确保数据类型一致 - 将所有输入转换为相同的数据类型
            dtype = word_embeds.dtype
            got_nodes = got_nodes.to(dtype=dtype)
            adj_matrix = adj_matrix.to(dtype=dtype)
            # 图结构编码
            graph_embeds = self.gat(got_nodes, adj_matrix)
            # 交叉注意力计算
            cross_attn_output, _ = self.cross_attn(
                query=word_embeds,
                key=graph_embeds,
                value=graph_embeds,
                need_weights=False
            )
            # 门控融合
            global_graph_info = graph_embeds.mean(dim=1, keepdim=True)
            gate_input = torch.cat([
                word_embeds,
                cross_attn_output,
                global_graph_info.expand(-1, word_embeds.size(1), -1)
            ], dim=-1)
            gate = torch.sigmoid(self.gate_linear(gate_input))
            fused_embeds = (1 - gate) * word_embeds + gate * cross_attn_output
            fused_embeds = self.dropout(self.output_linear(fused_embeds))
            return fused_embeds
        except Exception as e:
            print(f"图嵌入处理出错: {e}")
            # 发生错误时，回退到原始嵌入
            return word_embeds


class Qwen2GoTModel(Qwen2Model):
    def __init__(self, config):
        super(Qwen2GoTModel, self).__init__(config)
        self.gat = GAT(nfeat=config.hidden_size, nhid=config.hidden_size)
        self.gate = nn.Linear(2 * config.hidden_size, config.hidden_size)

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
            # 门控融合机制
            gate = torch.sigmoid(self.gate(
                torch.cat([word_embeds.mean(1), node_embeds.mean(1)], dim=-1)
            ))
            # 融合嵌入
            fused_embeds = (1 - gate).unsqueeze(1) * word_embeds + gate.unsqueeze(1) * node_embeds.unsqueeze(1)
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
            inputs_embeds=inputs_embeds,
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
