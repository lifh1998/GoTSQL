import os

import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import AutoConfig

from model import Qwen2GoTForCausalLM

model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"


def test_load_model_from_pretrained():
    print("\n===== Qwen2GoTForCausalLM 创建 =====")
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16
    # )
    model = Qwen2GoTForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float16,
        # quantization_config=quantization_config,
        # load_in_8bit=True,
    )
    return model


def test_load_model_by_config(model_name=model_name):
    config = AutoConfig.from_pretrained(model_name)
    config.torch_float16 = True
    model = Qwen2GoTForCausalLM(config)
    model_dict = model.state_dict()
    pretrained_dict = get_state_dict_from_safetensors()
    # 只加载匹配的参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model.load_state_dict(state_dict=pretrained_dict, strict=False)
    return model


def get_state_dict_from_safetensors():
    state_dict = {}
    model_dir = "C:\\Users\86177\.cache\huggingface\hub\models--Qwen--Qwen2.5-Coder-7B-Instruct\snapshots\c03e6d358207e414f1eca0bb1891e29f1db0e242"
    # model_dir = "/home/jamtc/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242"
    for i in range(1, 5):
        file_name = f"model-{i:05d}-of-00004.safetensors"
        file_path = os.path.join(model_dir, file_name)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
    return state_dict


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
    print("冻结的参数（requires_grad=False）：")
    for name, shape in frozen_params:
        print(f"参数名: {name}, 形状: {shape}")
    # print(f"冻结参数总数: {len(frozen_params)}\n")

    # 打印未冻结（可训练）的参数
    # print("未冻结的参数（requires_grad=True）：")
    # for name, shape in trainable_params:
    #     print(f"参数名: {name}, 形状: {shape}")
    # print(f"未冻结参数总数: {len(trainable_params)}")

# 执行测试
if __name__ == "__main__":
    model = test_load_model_by_config()
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
            "down_proj"  # Fully connected layer (down projection in MLP)
        ],
        lora_dropout=0.05,  # Dropout for LoRA layers
        bias="none",  # Bias handling
        task_type="CAUSAL_LM"  # Task type for causal language modeling
    )
    model = get_peft_model(model, lora_config)
    print(model)
    check_frozen_parameters(model)
    model.print_trainable_parameters()
