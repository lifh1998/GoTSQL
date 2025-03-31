import os

import torch
from accelerate import infer_auto_device_map, dispatch_model
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import AutoConfig, BitsAndBytesConfig

from model import Qwen2GoTForCausalLM

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"


def test_load_model_from_pretrained():
    print("\n===== Qwen2GoTForCausalLM 创建 =====")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = Qwen2GoTForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # quantization_config=quantization_config,
        # load_in_8bit=True,
    )
    return model


def test_load_model_by_config(model_name=model_name):
    config = AutoConfig.from_pretrained(model_name)
    config.torch_dtype = torch.float16
    model = Qwen2GoTForCausalLM(config)
    print(next(model.parameters()).dtype)
    # 加载预训练权重
    load_state_dict_from_safetensors(model=model, dtype=torch.float16)
    return model


def load_state_dict_from_safetensors(model, dtype=torch.float16):
    # model_dir = "C:\\Users\86177\.cache\huggingface\hub\models--Qwen--Qwen2.5-Coder-7B-Instruct\snapshots\c03e6d358207e414f1eca0bb1891e29f1db0e242"
    model_dir = "/home/jamtc/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242"
    keys = model.state_dict()

    # 使用进度条跟踪加载进度
    from tqdm import tqdm

    for i in tqdm(range(1, 5), desc="Loading model files"):
        state_dict = {}
        file_name = f"model-{i:05d}-of-00004.safetensors"
        file_path = os.path.join(model_dir, file_name)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                # 直接转换为所需精度，节省后续转换的内存
                tensor = f.get_tensor(k)
                if tensor.dtype == torch.float32:
                    tensor = tensor.to(dtype)
                state_dict[k] = tensor
        state_dict = {k: v for k, v in state_dict.items() if k in keys}
        model.load_state_dict(state_dict, strict=False)


def check_frozen_parameters(model):
    # 用于存储冻结和未冻结参数的信息
    frozen_params = []
    trainable_params = []

    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append((name, param.shape))
        else:
            frozen_params.append((name, param.shape))

    # 打印冻结的参数
    for name, shape in frozen_params:
        print(f"参数名: {name}, 形状: {shape}")
    # print(f"冻结参数总数: {len(frozen_params)}\n")

    # 打印未冻结（可训练）的参数
    # for name, shape in trainable_params:
    #     print(f"参数名: {name}, 形状: {shape}")
    # print(f"未冻结参数总数: {len(trainable_params)}")


def activate_partial_parameters(model):
    target_names = ['graph_fused']
    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        for target_name in target_names:
            if target_name in name:
                param.requires_grad = True


def test_get_peft_model(model):
    lora_config = LoraConfig(
        r=16,  # Rank of the low-rank matrices
        lora_alpha=32,  # Scaling factor
        target_modules=[
            "q_proj",  # Query projection
            "k_proj",  # Key projection
            "v_proj",  # Value projection
            "o_proj",  # Output projection (optional, if you want to include it)
            "gate_proj",  # Fully connected layer (gate projection in MLP)
            "up_proj",  # Fully connected layer (up projection in MLP)
            "down_proj",  # Fully connected layer (down projection in MLP)
            "lm_head",  # Output layer
        ],
        lora_dropout=0.05,  # Dropout for LoRA layers
        bias="none",  # Bias handling
        task_type="CAUSAL_LM"  # Task type for causal language modeling
    )
    model = get_peft_model(model, lora_config)
    activate_partial_parameters(model)
    # check_frozen_parameters(model)
    return model


# 执行测试
if __name__ == "__main__":
    # model = test_load_model_by_config()
    # model = test_get_peft_model(model)
    # print(model)
    # model.print_trainable_parameters()
    #
    # # 转换为半精度并分发到多个设备
    # model = model.to(torch.float16)
    # device_map = infer_auto_device_map(
    #     model,
    #     max_memory={0: "12GiB", 1: "16GiB"}  # 根据你的GPU配置调整
    # )
    # model = dispatch_model(model, device_map=device_map)

    model = Qwen2GoTForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model.post_init()
    for name, param in model.named_parameters():
        print(f"name:{name}, param:{param}")
