#!/bin/bash

python create_model.py \
    --qwen2_vl_model_id "Qwen/Qwen2-VL-2B-Instruct" \
    --attn_implementation "flash_attention_2" \
    --save_local \
    --output_dir "Qwen2-VL-2B-Instruct-Audio" \
    --push_to_hub \
    --create_new_repo \
    --push_to_hub_repo_id "<REPO_ID>"
