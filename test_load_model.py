from safetensors import safe_open
import torch
import os

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
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


def test_load_model_by_config():
    config = AutoConfig.from_pretrained(model_name)
    config.torch_dtype = torch.float16
    model = Qwen2GoTForCausalLM(config)
    model_dict = model.state_dict()
    pretrained_dict = get_state_dict_from_safetensors()
    # 只加载匹配的参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model.load_state_dict(state_dict=pretrained_dict, strict=False)
    return model

def get_state_dict_from_safetensors():
    state_dict = {}
    model_dir = "C:\Users\86177\.cache\huggingface\hub\models--Qwen--Qwen2.5-Coder-7B-Instruct\snapshots\c03e6d358207e414f1eca0bb1891e29f1db0e242"
    for i in range(1, 5):
        file_name = f"model-{i:05d}-of-00004.safetensors"
        file_path = os.path.join(model_dir, file_name)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
    return state_dict


# 执行测试
if __name__ == "__main__":
    model = test_load_model_by_config()
    print(model.state_dict())