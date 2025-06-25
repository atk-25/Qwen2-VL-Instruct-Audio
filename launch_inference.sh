#!/bin/bash

python inference.py \
    --base_model_id "<base_model_id>" \
    --lora_adapters_id "<lora_adapters_id>" \
    --use_quantized_model \
    --qlora_model_id "<qlora_model_id>" \
    --repos_private \
    --attn_implementation "flash_attention_2" \
    --per_device_inference_batch_size 2 \
    --max_new_tokens 256
