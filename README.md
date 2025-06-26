## Introduction

## Installations:
```
pip install -q torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install trl datasets bitsandbytes peft wandb accelerate matplotlib IPython librosa evaluate --upgrade evaluate jiwer
```
```
## install transformers from this repo
pip install git+https://github.com/atk-25/transformers.git
```
```
## install qwen-vl-utils from this repo
git clone https://github.com/atk-25/Qwen2.5-VL.git
cd Qwen2.5-VL/qwen-vl-utils
pip install .
```
```
## Install FlashAttention-2 (Optional):
pip install flash-attn
```

# Setup:
1. Create model: to run create_model.py use the following:
   ```
   python create_model.py \
       --qwen2_vl_model_id "Qwen/Qwen2-VL-2B-Instruct" \
       --attn_implementation "flash_attention_2" \
       --save_local \
       --output_dir "Qwen2-VL-2B-Instruct-Audio" \
       --push_to_hub \
       --create_new_repo \
       --push_to_hub_repo_id "<REPO_ID>"
   ```
   or you can also use the shell script:
   ```
   bash launch_create_model.sh
   ```
3. Pretrain the audio projector:
   ```
   bash launch_pretrain_audio_projector.sh
   ```
4. Finetune using QLoRA:
   ```
   bash launch_finetune_qlora.sh
   ```
6. Evaluate finetuned model:
   ```
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
   ```
   or use the shell script:
   ```
   bash launch_evaluate_model.sh
   ```
8. run inference:
   ```
   bash launch_inference.sh
   ```
