import os
import warnings

import torch
# import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback

from model import GoTSQLModelForCausalLM
from process_dataset import generate_complete_graph_data, process_seq_data, process_graph_data, custom_data_collator

warnings.filterwarnings("ignore",
                        message="Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`.")


# class WandbCallback(TrainerCallback):
#     def on_log(self, args, state, control, logs=None, **kwargs):
#         if logs is not None:
#             wandb.log(logs)


def test_train():
    # 0. Load model and tokenizer
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GoTSQLModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
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
            # "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj",
            # "lm_head"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    def activate_partial_parameters(model):
        target_names = ['graph_fused']
        for name, param in model.named_parameters():
            for target_name in target_names:
                if target_name in name:
                    param.requires_grad = True

    activate_partial_parameters(model)
    model.print_trainable_parameters()  # Optional: Check which parameters are trainable

    # 2. Prepare data
    # 序列数据编码
    def preprocess_seq_data(examples):
        return process_seq_data(examples, tokenizer)

    # 图数据编码
    def preprocess_graph_data(examples):
        return process_graph_data(examples, tokenizer)

    dataset = Dataset.from_json("/home/jamtc/paper-code/lifh/my-experiments/gotsql/datasets/train/train_spider.json")
    # 应用预处理
    dataset = dataset.map(generate_complete_graph_data, batched=True)
    dataset = dataset.map(preprocess_seq_data, batched=True)
    dataset = dataset.map(preprocess_graph_data, batched=True)

    # 设置数据集格式为 PyTorch
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels", "got_nodes", "adj_matrix"], device="cpu")

    # 3. Define training arguments
    training_args = TrainingArguments(
        output_dir="./output/model",
        logging_dir="./output/log",
        logging_strategy="epoch",
        save_strategy="epoch",
        gradient_checkpointing=True,
        label_names=["labels"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        learning_rate=2e-05,
        weight_decay=0.01,
        fp16=True,
        warmup_steps=500,
        # deepspeed="./config/ds_config.json",
    )

    # 4. Initialize Trainer
    def data_collator(features):
        return custom_data_collator(features)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # eval_dataset=dataset.shuffle(seed=42).select(range(200)),
        data_collator=data_collator,
        # callbacks=[WandbCallback()],
    )

    # 5. Train model
    trainer.train()

    # 6. Save model
    trainer.save_model()


if __name__ == '__main__':
    os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 明确指定使用的GPU
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ["WANDB_API_KEY"] = "0809aa4ed98dbbba4c94164e8cee38d82184c279"

    # wandb.login()
    # # 配置 W&B
    # wandb.init(
    #     project="qwen2got-training",  # 项目名称，可以自定义
    #     config={  # 记录超参数
    #         "model_name": "Qwen/Qwen2.5-Coder-3B-Instruct",
    #         "lora_r": 16,
    #         "lora_alpha": 32,
    #         "learning_rate": 2e-5,
    #         "num_epochs": 3,
    #         "batch_size": 8,
    #         "gradient_accumulation_steps": 4
    #     }
    # )

    test_train()
