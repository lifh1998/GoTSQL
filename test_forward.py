import torch
from transformers import AutoTokenizer
from test_load_model import test_load_model_by_config

def test_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建模型
    model = test_load_model_by_config().to(device)
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    # 处理输入数据
    texts = ['张三和李四关系好吗？', '王五是兄弟吗']
    node_names = [['张三', '李四', '王五'], ['张三', '李四', '王五']]
    adj_matrices = [[[1, 0, 0], [1, 1, 0], [0, 1, 1]], [[1, 0, 0], [1, 1, 0], [0, 1, 1]]]
    # 对文本进行分词
    text_encodings = tokenizer(texts, padding=True, return_tensors="pt")
    # 对节点名称进行分词和embedding处理
    node_embeddings = []
    for batch_nodes in node_names:
        batch_embeds = []
        for node in batch_nodes:
            # 获取每个节点的token ID
            node_encoding = tokenizer(node, return_tensors="pt", padding=False)
            # 获取嵌入 (简化处理，实际使用需确保模型加载了预训练权重)
            with torch.no_grad():
                node_embed = model.model.embed_tokens(node_encoding['input_ids']).mean(dim=1)  # 平均池化
            batch_embeds.append(node_embed)
        # 将该批次的节点嵌入堆叠起来
        batch_embeds = torch.cat(batch_embeds, dim=0)
        node_embeddings.append(batch_embeds)
    # 将节点嵌入填充到相同长度并转换为张量
    max_nodes = max(len(nodes) for nodes in node_names)
    padded_node_embeddings = []
    for batch_embeds in node_embeddings:
        if batch_embeds.shape[0] < max_nodes:
            padding = torch.zeros(max_nodes - batch_embeds.shape[0], batch_embeds.shape[1])
            batch_embeds = torch.cat([batch_embeds, padding], dim=0)
        padded_node_embeddings.append(batch_embeds)
    # 转换为模型所需的批次格式
    got_nodes = torch.stack(padded_node_embeddings)
    # 处理邻接矩阵
    adj_matrix = torch.tensor(adj_matrices)
    # 前向传播
    with torch.no_grad():
        outputs = model(
            input_ids=text_encodings['input_ids'].to(device),
            attention_mask=text_encodings['attention_mask'].to(device),
            got_nodes=got_nodes.to(device),
            adj_matrix=adj_matrix.to(device),
        )
        # 假设模型的输出是生成的 token IDs
        generated_ids = outputs.logits.argmax(dim=-1)  # 获取生成的 token ID
        # 解码生成的 token IDs
        decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(f"输出: {decoded_outputs}")
        return decoded_outputs


if __name__ == "__main__":
    outputs = test_forward()
