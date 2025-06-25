#!/bin/bash

# "atk-25/Qwen2-VL-2B-Instruct-Audio-qlora"
# --use_quantized_model \
# --qlora_model_id "atk-25/Qwen2-VL-2B-Instruct-Audio-qlora" \

python evaluate_model.py \
    --base_model_id "atk-25/Qwen2-VL-2B-Instruct-Audio" \
    --lora_adapters_id "atk-25/Qwen2-VL-2B-Instruct-Audio-LoRA-Adapters" \
    --use_quantized_model \
    --qlora_model_id "atk-25/Qwen2-VL-2B-Instruct-Audio-qlora" \
    --repos_private \
    --attn_implementation "flash_attention_2" \
    --per_device_eval_batch_size 2 \
    --number_samples_for_eval 32 \
    --max_new_tokens 256