#!/bin/bash

python evaluate_model.py \
    --base_model_id "<BASE_MODEL_ID>" \
    --lora_adapters_id "<LORA_ADAPTERS_ID>" \
    --use_quantized_model \
    --qlora_model_id "<QLORA_MODEL_ID>" \
    --repos_private \
    --attn_implementation "flash_attention_2" \
    --per_device_eval_batch_size 8 \
    --number_samples_for_eval 128 \
    --max_new_tokens 256
