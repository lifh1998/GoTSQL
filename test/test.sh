#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name="Qwen2.5-Coder-3B-Instruct"
output_dir="./output"

python test_read_arguments.py \
    --model_name_or_path "${model_name}" \
    --do_train \
    --do_eval \
    --lora_alpha 16 \
    --lora_r 8 \
    --target_modules "q_proj" "k_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj" "lm_head" \
    --lora_dropout 0.05 \
    --output_dir "${OUTPUT_DIR}/model" \
    --logging_dir "${OUTPUT_DIR}/log" \
    --logging_strategy "epoch" \
    --save_strategy "epoch" \
    --gradient_checkpointing \
    --deepspeed "./ds_config.json" \
    --seed 0
