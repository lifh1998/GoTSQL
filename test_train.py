import os

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig, TrainerCallback

from model import Qwen2GoTForCausalLM

import warnings
warnings.filterwarnings("ignore", message="Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.")

class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            wandb.log(logs)

def test_train():
    # 0. Load model and tokenizer
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = Qwen2GoTForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        # torch_dtype=torch.float32
        # quantization_config=quantization_config,
        # low_cpu_mem_usage=True,
    )
    model.post_init()

    # 1. Apply LoRA to the model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Optional: Check which parameters are trainable

    def activate_partial_parameters(model):
        target_names = ['graph_fused']
        for name, param in model.named_parameters():
            for target_name in target_names:
                if target_name in name:
                    param.requires_grad = True

    activate_partial_parameters(model)

    # 2. Prepare data
    texts = [
        '你好,很高兴认识你！',
        '请写一个学生课程成绩查询的SQL语句。'
    ]
    responses = [
        '你好！我是Claude，有什么我可以帮助你的吗？',
        'SELECT s.student_name, c.course_name, sc.score FROM students s JOIN student_course sc ON s.student_id = sc.student_id JOIN courses c ON sc.course_id = c.course_id WHERE s.student_id = [学生ID];'
    ]
    node_names = [
        ['张三', '李四', '王五'],
        ['张三', '李四', '王五', '赵六']
    ]
    adj_matrices = [
        [[1, 0, 0], [1, 1, 0], [0, 1, 1]],
        [[1, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 1], [1, 0, 0, 1]]
    ]

    # 创建数据字典
    data_dict = {
        "input_text": texts,
        "response_text": responses,
        "node_names": node_names,
        "adj_matrices": adj_matrices
    }
    dataset = Dataset.from_dict(data_dict)

    # 定义预处理函数
    def preprocess_function(examples):
        # 对输入和响应进行 tokenization，启用 padding 和 truncation
        input_tokens = tokenizer(
            examples["input_text"],
            padding=True,  # 填充到最长序列
            truncation=True,  # 截断超长序列
            return_tensors="pt"
        )
        response_tokens = tokenizer(
            examples["response_text"],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # 移除多余的维度（batch 维度）
        result = {
            "input_ids": input_tokens["input_ids"],
            "attention_mask": input_tokens["attention_mask"],
            "response_ids": response_tokens["input_ids"],
            "response_mask": response_tokens["attention_mask"],
            "node_names": examples["node_names"],
            "adj_matrices": examples["adj_matrices"]
        }
        return result

    # 应用预处理
    dataset = dataset.map(preprocess_function, batched=True)

    # 将数据转换为训练所需格式
    def format_function(examples):
        input_ids = examples["input_ids"]
        attention_mask = examples["attention_mask"]
        response_ids = examples["response_ids"]
        response_mask = examples["response_mask"]

        # 将列表转换为张量
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        response_ids = torch.tensor(response_ids)
        response_mask = torch.tensor(response_mask)

        # 拼接 input 和 response
        labels = torch.cat([
            torch.full((input_ids.shape[0], input_ids.shape[1]), -100, dtype=torch.long),
            response_ids
        ], dim=1)
        full_input_ids = torch.cat([input_ids, response_ids], dim=1)
        full_attention_mask = torch.cat([attention_mask, response_mask], dim=1)

        # 处理图数据
        node_ids = [
            [tokenizer(node, return_tensors="pt", padding=False)["input_ids"].squeeze(0) for node in nodes]
            for nodes in examples["node_names"]
        ]

        # 计算最大节点 token 长度和最大节点数量
        max_node_len = max(max(ids.size(0) for ids in node_batch) for node_batch in node_ids)
        max_num_nodes = max(len(node_batch) for node_batch in node_ids)

        # 填充节点数据
        padded_node_ids = []
        for node_batch in node_ids:
            padded_nodes = [
                torch.cat([ids, torch.zeros(max_node_len - ids.size(0), dtype=torch.long)]) if ids.size(
                    0) < max_node_len else ids
                for ids in node_batch
            ]
            while len(padded_nodes) < max_num_nodes:
                padded_nodes.append(torch.zeros(max_node_len, dtype=torch.long))
            padded_nodes = padded_nodes[:max_num_nodes]
            padded_node_ids.append(torch.stack(padded_nodes))

        got_nodes = torch.stack(padded_node_ids)

        # 填充邻接矩阵
        adj_matrices = examples["adj_matrices"]
        padded_adj_matrices = []
        for adj in adj_matrices:
            adj_tensor = torch.tensor(adj, dtype=torch.float)
            n = adj_tensor.size(0)
            if n < max_num_nodes:
                padded_adj = torch.zeros((max_num_nodes, max_num_nodes), dtype=torch.float)
                padded_adj[:n, :n] = adj_tensor
                padded_adj_matrices.append(padded_adj)
            else:
                padded_adj_matrices.append(adj_tensor[:max_num_nodes, :max_num_nodes])
        adj_matrix = torch.stack(padded_adj_matrices)

        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "labels": labels,
            "got_nodes": got_nodes,
            "adj_matrix": adj_matrix
        }

    dataset = dataset.map(format_function, batched=True)

    # 设置数据集格式为 PyTorch
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "got_nodes", "adj_matrix"])

    # 3. Define training arguments
    training_args = TrainingArguments(
        output_dir="./output/model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        logging_dir="./output/log",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        fp16=True,
        label_names=["labels"],
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        # local_rank=-1,
        # ddp_find_unused_parameters=False,
        # ddp_bucket_cap_mb=25,
        # dataloader_num_workers=2,
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[WandbCallback()],
    )

    # 5. Train model
    trainer.train()

    # 6. Save model
    # trainer.save_model()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_API_KEY"] = "0809aa4ed98dbbba4c94164e8cee38d82184c279"

    wandb.login()
    # 配置 W&B
    wandb.init(
        project="qwen2got-training",  # 项目名称，可以自定义
        config={  # 记录超参数
            "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "batch_size": 1,
            "gradient_accumulation_steps": 4
        }
    )

    test_train()
