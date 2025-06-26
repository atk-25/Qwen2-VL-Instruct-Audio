import os
import argparse
import logging
import gc
import time
import torch
import torch.nn as nn
from huggingface_hub import HfApi, login
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers import WhisperForConditionalGeneration
from transformers.utils import is_flash_attn_2_available


logger = logging.getLogger(__name__)


def arg_parser():

    parser = argparse.ArgumentParser()

    attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

    parser.add_argument("--qwen2_vl_model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--attn_implementation", type=str, default=attn_implementation)
    parser.add_argument("--save_local", action="store_true")
    parser.add_argument("--output_dir", type=str, default="Qwen2-VL-2B-Instruct-Audio")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--create_new_repo", action="store_true")
    parser.add_argument("--push_to_hub_repo_id", type=str, default=None)

    args = parser.parse_args()

    return args


def init_weights(module):
    if isinstance(module, nn.Linear):
        std = model.config.get_text_config().initializer_range
        module.weight.data.normal_(mean=0.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()


def clear_memory():
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'model_whisper' in globals(): del globals()['model_whisper']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


if __name__ == "__main__":

    args = arg_parser()

    if args.push_to_hub or args.create_new_repo:
        assert os.environ.get('HF_TOKEN') is not None, "Set up HF_TOKEN if you want to push model to a HF repo"
        assert args.push_to_hub_repo_id is not None, "push_to_hub_repo_id is not provided"
        HF_TOKEN = os.environ.get('HF_TOKEN')
        login(token=HF_TOKEN)

    if args.save_local:
        assert args.output_dir is not None, "output_dir is not provided"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # download Qwen2-VL model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.qwen2_vl_model_id, torch_dtype=torch.bfloat16, device_map=device, attn_implementation=args.attn_implementation
    )
    processor = Qwen2VLProcessor.from_pretrained(args.qwen2_vl_model_id)

    ### reinitialize audio_projector weights
    model.audio.audio_projector.ln.apply(init_weights)
    model.audio.audio_projector.mlp.apply(init_weights)

    # load pretrained weights from whisper-large-v3-turbo into the audio encoder
    model_whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo",
                                                                    torch_dtype=torch.bfloat16, device_map=device)
    model.audio.audio_encoder.load_state_dict(model_whisper.model.encoder.state_dict())

    # Save model in local directory
    if args.save_local:
        model.save_pretrained(args.output_dir)
        print("model with randomly initialized audio projector module saved to local directory.")

    ### create a repository
    if args.create_new_repo:
        api = HfApi()
        api.create_repo(repo_id=args.push_to_hub_repo_id, private=True)
        logger.info(
            f"New repo created for pushing model with randomly initialized audio projector module, repo_id: {args.push_to_hub_repo_id}")
        print(f"New repo created for pushing model with randomly initialized audio projector module, repo_id: {args.push_to_hub_repo_id}")

    # push processor and model to repo
    if args.push_to_hub:
        processor.push_to_hub(args.push_to_hub_repo_id)
        model.push_to_hub(args.push_to_hub_repo_id)
        logger.info(
            f"model with randomly initialized audio projector module and processor pushed to repo_id: {args.push_to_hub_repo_id}")
        print(f"model with randomly initialized audio projector module and processor pushed to repo_id: {args.push_to_hub_repo_id}")

    clear_memory()
