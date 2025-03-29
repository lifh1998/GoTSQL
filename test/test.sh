#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

model_name="Qwen2.5-Coder-3B-Instruct"
output_dir="./output"

deepspeed --num_gpus=2 test_read_arguments.py \
    --model_name_or_path "${model_name}" \
    --tokenizer_name "${model_name}" \
    --do_train \
    --do_eval \
    --lora_r 8 \
    --lora_alpha 16 \
    --target_modules "q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj" "lm_head" \
    --lora_dropout 0.05 \
    --output_dir "${output_dir}/model" \
    --logging_dir "${output_dir}/log" \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --gradient_checkpointing \
    --deepspeed "../config/ds_config.json" \
    --seed 0
