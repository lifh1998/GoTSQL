import os

import torch
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig

from model import Qwen2GoTForCausalLM


def test_train():
    accelerator = Accelerator()

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
        torch_dtype=torch.float16
        # quantization_config=quantization_config,
        # low_cpu_mem_usage=True,
    )
    model.post_init()

    # 1. Apply LoRA to the model
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
    model.print_trainable_parameters()  # Optional: Check which parameters are trainable

    def activate_partial_parameters(model):
        target_names = ['embed_fused']
        for name, param in model.named_parameters():
            for target_name in target_names:
                if target_name in name:
                    param.requires_grad = True

    activate_partial_parameters(model)

    # 2. Prepare data (unchanged from your original code)
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
        ['张三', '李四', '王五']
    ]
    adj_matrices = [
        [[1, 0, 0], [1, 1, 0], [0, 1, 1]],
        [[1, 0, 0], [1, 1, 0], [0, 1, 1]]
    ]

    # Dataset class (unchanged)
    class GraphTextDataset(Dataset):
        def __init__(self, inputs, responses, tokenizer, node_names=None, adj_matrices=None):
            self.tokenizer = tokenizer
            self.inputs = inputs
            self.responses = responses
            self.has_graph_data = node_names is not None and adj_matrices is not None
            self.node_names = node_names if self.has_graph_data else []
            self.adj_matrices = adj_matrices if self.has_graph_data else []

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            input_text = self.inputs[idx]
            response_text = self.responses[idx]

            input_tokens = self.tokenizer(input_text, return_tensors="pt", padding=False)
            response_tokens = self.tokenizer(response_text, return_tensors="pt", padding=False)

            input_ids = input_tokens["input_ids"].squeeze(0)
            attention_mask = input_tokens["attention_mask"].squeeze(0)

            response_ids = response_tokens["input_ids"].squeeze(0)
            response_mask = response_tokens["attention_mask"].squeeze(0)

            labels = torch.cat([
                torch.full((input_ids.size(0),), -100, dtype=torch.long),
                response_ids
            ])

            full_input_ids = torch.cat([input_ids, response_ids])
            full_attention_mask = torch.cat([attention_mask, response_mask])

            result = {
                "input_ids": full_input_ids,
                "attention_mask": full_attention_mask,
                "labels": labels
            }

            if self.has_graph_data:
                node_ids = [self.tokenizer(node, return_tensors="pt", padding=False)["input_ids"].squeeze(0) for node in
                            self.node_names[idx]]
                max_node_len = max(ids.size(0) for ids in node_ids)
                padded_node_ids = [
                    torch.cat([ids, torch.zeros(max_node_len - ids.size(0), dtype=torch.long)]) if ids.size(
                        0) < max_node_len else ids for ids in node_ids]
                got_nodes = torch.stack(padded_node_ids)
                adj_matrix = torch.tensor(self.adj_matrices[idx], dtype=torch.float)
                result["got_nodes"] = got_nodes
                result["adj_matrix"] = adj_matrix

            return result

    # Collate function (unchanged)
    def collate_fn(batch):
        max_input_len = max(item["input_ids"].size(0) for item in batch)
        max_attn_len = max(item["attention_mask"].size(0) for item in batch)
        max_label_len = max(item["labels"].size(0) for item in batch)

        padded_inputs = []
        padded_attention = []
        padded_labels = []

        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]

            if input_ids.size(0) < max_input_len:
                padding = torch.zeros(max_input_len - input_ids.size(0), dtype=torch.long)
                input_ids = torch.cat([input_ids, padding])
            if attention_mask.size(0) < max_attn_len:
                padding = torch.zeros(max_attn_len - attention_mask.size(0), dtype=torch.long)
                attention_mask = torch.cat([attention_mask, padding])
            if labels.size(0) < max_label_len:
                padding = torch.full((max_label_len - labels.size(0),), -100, dtype=torch.long)
                labels = torch.cat([labels, padding])

            padded_inputs.append(input_ids)
            padded_attention.append(attention_mask)
            padded_labels.append(labels)

        batch_dict = {
            "input_ids": torch.stack(padded_inputs),
            "attention_mask": torch.stack(padded_attention),
            "labels": torch.stack(padded_labels)
        }

        if "got_nodes" in batch[0]:
            max_nodes = max(item["got_nodes"].size(0) for item in batch)
            max_node_len = max(item["got_nodes"].size(1) for item in batch)

            padded_nodes = []
            padded_adj = []

            for item in batch:
                nodes = item["got_nodes"]
                adj = item["adj_matrix"]

                if nodes.size(0) < max_nodes or nodes.size(1) < max_node_len:
                    padded = torch.zeros(max_nodes, max_node_len, dtype=torch.long)
                    padded[:nodes.size(0), :nodes.size(1)] = nodes
                    nodes = padded
                if adj.size(0) < max_nodes or adj.size(1) < max_nodes:
                    padded = torch.zeros(max_nodes, max_nodes, dtype=torch.float)
                    padded[:adj.size(0), :adj.size(1)] = adj
                    adj = padded

                padded_nodes.append(nodes)
                padded_adj.append(adj)

            batch_dict["got_nodes"] = torch.stack(padded_nodes)
            batch_dict["adj_matrix"] = torch.stack(padded_adj)

        return batch_dict

    # Create dataset
    dataset = GraphTextDataset(texts, responses, tokenizer, node_names, adj_matrices)

    # 3. Define training arguments
    training_args = TrainingArguments(
        output_dir="./outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=5e-5,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        # 多GPU训练相关设置
        local_rank=-1,  # 让Trainer自动检测分布式环境
        ddp_find_unused_parameters=False,  # 优化DDP性能
        ddp_bucket_cap_mb=25,
        dataloader_num_workers=2,  # 数据加载器的工作进程数
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )
    # 4.1 Use accelerator
    trainer.model, trainer.optimizer, trainer.train_dataset, trainer.data_collator = accelerator.prepare(
        trainer.model, trainer.optimizer, trainer.train_dataset, trainer.data_collator
    )

    # 5. Train model
    trainer.train()
    accelerator.wait_for_everyone()  # Sync all processes
    if accelerator.is_main_process:
        model.save_pretrained("./qwen2got_lora_model")
        tokenizer.save_pretrained("./qwen2got_lora_model")

    # 6. Save the LoRA-adapted model
    model.save_pretrained("./qwen2got_lora_model")
    tokenizer.save_pretrained("./qwen2got_lora_model")


if __name__ == '__main__':
    # 设置要使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 指定使用CUDA 0和CUDA 1
    test_train()
