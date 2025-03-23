import types

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import QwenGoTEnhancedEmbeddings


def main():
    # 测试配置参数
    MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
    BATCH_SIZE = 2
    SEQ_LENGTH = 32
    NODE_NUM = 10

    # 1. 加载分词器
    print("加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    # 2. 加载模型 - 不使用accelerate的device_map
    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        # 不使用device_map="auto"，而是明确加载到单个设备
        low_cpu_mem_usage=True
    )

    # 将整个模型移动到GPU或CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(model)

    # 确定模型的实际hidden_size
    HIDDEN_SIZE = model.config.hidden_size
    print(f"模型维度: {HIDDEN_SIZE}")

    # 3. 保存原始embeddings
    original_embeddings = model.model.embed_tokens

    # 4. 创建自定义嵌入层
    print("创建自定义嵌入层...")
    custom_embeddings = QwenGoTEnhancedEmbeddings(
        config=model.config,
        original_embeddings=original_embeddings
    ).to(device).to(torch.float16)  # 确保嵌入层在正确的设备和数据类型上

    # 替换嵌入层
    model.model.embed_tokens = custom_embeddings

    # 5. 包装前向传播 - 修改为使用monkey patch形式，避免内存问题
    original_forward = model.forward

    def forward_wrapper(self, input_ids=None, attention_mask=None, got_nodes=None, adj_matrix=None, **kwargs):
        # 如果提供了图结构数据，手动处理嵌入
        if got_nodes is not None and adj_matrix is not None and input_ids is not None:
            with torch.no_grad():  # 避免不必要的梯度计算
                # 确保数据类型一致
                dtype = next(self.parameters()).dtype
                got_nodes = got_nodes.to(device=device, dtype=dtype)
                adj_matrix = adj_matrix.to(device=device, dtype=dtype)

                # 手动调用嵌入层
                inputs_embeds = self.model.embed_tokens(input_ids, got_nodes, adj_matrix)
                # 使用生成的嵌入替代input_ids
                kwargs['inputs_embeds'] = inputs_embeds
                input_ids = None

        # 调用原始forward方法
        return original_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    # 替换forward方法
    model.forward = types.MethodType(forward_wrapper, model)

    # 6. 准备模拟输入数据
    print("准备输入数据...")
    texts = ["Hello, how are you?", "I'm fine, thank you!"]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 创建图结构输入 - 确保统一数据类型
    dtype = next(model.parameters()).dtype
    got_nodes = torch.randn(BATCH_SIZE, NODE_NUM, HIDDEN_SIZE, device=device, dtype=dtype)
    adj_matrix = torch.randint(0, 2, (BATCH_SIZE, NODE_NUM, NODE_NUM), device=device).to(dtype)

    # 7. 使用try-except包装测试以防止程序崩溃
    print("测试标准输入...")
    try:
        with torch.no_grad():
            standard_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
        print(f"标准输入成功，logits shape: {standard_outputs.logits.shape}")
    except Exception as e:
        print(f"标准输入测试失败: {e}")

    print("测试图结构输入...")
    try:
        with torch.no_grad():
            graph_outputs = model(
                input_ids=inputs["input_ids"],
                got_nodes=got_nodes,
                adj_matrix=adj_matrix,
                attention_mask=inputs["attention_mask"]
            )
        print(f"图结构输入成功，logits shape: {graph_outputs.logits.shape}")
    except Exception as e:
        print(f"图结构输入测试失败: {e}")


if __name__ == "__main__":
    main()
