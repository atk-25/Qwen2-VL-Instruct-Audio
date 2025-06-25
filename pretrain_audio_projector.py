# TASK: Load the Qwen2VL checkpoint into your model, load whisper turbo, save and push to your repo -  Now, the model and tokenizer should both be there.

import os
import argparse
import logging
import gc
import time
import torch
from huggingface_hub import HfApi, login
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_audio_info
from transformers.utils import is_flash_attn_2_available
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from evaluate import load
import wandb


logger = logging.getLogger(__name__)


def arg_parser():

    parser = argparse.ArgumentParser()

    attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=attn_implementation)
    parser.add_argument("--save_local", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--create_new_repo", action="store_true")
    parser.add_argument("--push_to_hub_repo_id", type=str, default=None)
    parser.add_argument("--repos_private", action="store_true")

    ### SFTConfig related arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gradient_checkpointing_kwargs", type=dict, default={"use_reentrant": False})
    parser.add_argument("--max_seq_length", type=int, default=2048)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--optim", type=str, default="adamw_torch_fused")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")

    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.001)

    # logging parameters
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    parser.add_argument("--output_dir", type=str, default="Checkpoints")

    parser.add_argument("--dataset_kwargs", type=dict, default={"skip_prepare_dataset": True})

    # wandb project and name associated with this run
    parser.add_argument("--WANDB_PROJECT", type=str, default=None)
    parser.add_argument("--WANDB_NAME", type=str, default=None)

    # max_new_tokens:   used for evaluation function
    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()

    return args


system_message = """You are a Vision Language and Audio Model specialized in interpreting visual and audio data.
Your task is to transcribe the provided audio signal.
Focus on delivering accurate, succinct answers based on the audio information. Avoid additional explanation unless absolutely necessary."""

def create_conversation_template_for_training(example):

    audio = example["wav"]
    text = example["text"]
    prompt = "Transcribe this audio."
    assistant_response = f"{text}"

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio": audio,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": assistant_response,
                }
            ],
        },
    ]

    return messages


def get_dataset():

    train_dataset = load_dataset("speechbrain/LargeScaleASR", data_files=["small/train-0000*", "small/train-0001*"],
                                 num_proc=12)
    test_dataset = load_dataset("speechbrain/LargeScaleASR", data_files=["test/test-00000*"], num_proc=12)
    train_dataset = train_dataset["train"]
    test_dataset = test_dataset["train"]
    test_dataset = test_dataset.select(range(100))  # only 100 samples used for accelerated testing
    logger.info(f"train_dataset length: {len(train_dataset)}")
    logger.info(f"test_dataset length: {len(test_dataset)}")

    train_dataset = train_dataset.to_list()
    test_dataset = test_dataset.to_list()

    return train_dataset, test_dataset


def get_model_and_processor(args, device):

    # download Qwen2-VL model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map=device, attn_implementation=args.attn_implementation
    )
    processor = Qwen2VLProcessor.from_pretrained(args.model_id)

    return model, processor


def collate_fn(examples):

    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    audios = [process_audio_info(example)[0][0] for example in examples]
    _, audio_sampling_rate = process_audio_info(examples[0])

    # Get batch inputs to LLM by tokenizing the texts and processing the audios
    batch = processor(text=texts, audios=audios, audio_sampling_rate=audio_sampling_rate, return_tensors="pt",
                      padding=True)  # a list of dictionaries with these keys ['input_ids', 'attention_mask', 'audio_features']

    # Get the labels to the input_ids
    labels = batch["input_ids"].clone()
    # Mask the padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100  # pad_token_id = 151643

    ## Mask tokens related to audio
    audio_token_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)  # audio_token_id = 151658
    # Also include tokens that represent the start and end of audio tokens
    audio_tokens = [audio_token_id] + [151657, 151659]  # <|audio_start|> = 151657, <|audio_end|> = 151659

    for audio_token in audio_tokens:
        labels[labels == audio_token] = -100

    # add the labels to the input batch
    batch["labels"] = labels  # batch: a list of dictionaries with these keys ['input_ids', 'attention_mask', 'image_features', 'labels']

    return batch


def pretrain_audio_projector(model, processor, train_dataset, eval_dataset, args):

    for param in model.parameters():
        param.requires_grad = False
    for param in model.audio.audio_projector.parameters():
        param.requires_grad = True

    sft_config = SFTConfig(

        per_device_train_batch_size = args.per_device_train_batch_size,
        per_device_eval_batch_size = args.per_device_eval_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        gradient_checkpointing = args.gradient_checkpointing,
        gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
        max_seq_length = args.max_seq_length,

        num_train_epochs = args.num_train_epochs,
        max_steps = args.max_steps,
        learning_rate = args.learning_rate,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,

        bf16=args.bf16,
        tf32=args.tf32,

        warmup_steps = args.warmup_steps,
        weight_decay = args.weight_decay,

        # logging parameters
        logging_steps = args.logging_steps,
        eval_strategy = args.eval_strategy,
        eval_steps = args.eval_steps,
        save_strategy = args.save_strategy,
        save_steps = args.save_steps,
        report_to = args.report_to,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        output_dir = args.output_dir,

        dataset_kwargs=args.dataset_kwargs,

    )

    ### set up wandb.init
    wandb.init(
        project = args.WANDB_PROJECT,
        name = args.WANDB_NAME,
        config = sft_config,
    )

    trainer = SFTTrainer(
        model = model,
        args = sft_config,
        data_collator=collate_fn,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        processing_class = processor,
    )

    logger.info("Training Started.")

    trainer.train()

    logger.info("Training completed.")


def run_inference_asr(model, processor, device, examples, inference_batch_size=8, max_new_tokens=256, return_references=False):
    references = []
    output_texts = []
    num_examples = len(examples)
    batch_size = inference_batch_size
    num_batches = num_examples // batch_size + (1 if num_examples % batch_size != 0 else 0)

    ### run inference by iterating over batches
    start_idx = 0
    for i in range(num_batches):
        if start_idx + batch_size <= num_examples:
            end_idx = start_idx + batch_size
        else:
            end_idx = start_idx + num_examples % batch_size

        examples_batch = examples[start_idx:end_idx]

        references_batch = [example[2]["content"][0]["text"] for example in examples_batch]

        # Get the texts and audios
        if isinstance(examples_batch[0], dict):
            examples_batch = [examples_batch]
        texts = [processor.apply_chat_template(example[0:2], tokenize=False, add_generation_prompt=True) for example in
                 examples_batch]
        audios = [process_audio_info(example)[0][0] for example in examples_batch]
        _, audio_sampling_rate = process_audio_info(examples_batch[0])

        inputs = processor(text=texts, audios=audios, audio_sampling_rate=audio_sampling_rate, return_tensors="pt",
                           padding=True)  # a list of dictionaries with these keys ['input_ids', 'attention_mask', 'audio_features']

        inputs = inputs.to(device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts_batch = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        references.extend(references_batch)
        output_texts.extend(output_texts_batch)
        start_idx = end_idx

    if return_references:
        return output_texts, references
    else:
        return output_texts


def evaluate_model_wer(model, processor, device, eval_dataset, eval_batch_size=8, max_new_tokens=256):
    wer_metric = load("wer")
    output_texts, references = run_inference_asr(model, processor, device, examples=eval_dataset,
                                                  inference_batch_size=eval_batch_size, max_new_tokens=max_new_tokens, return_references=True)
    wer = 100 * wer_metric.compute(references=references, predictions=output_texts)

    return wer


def clear_memory():
    # Delete variables if they exist in the current global scope
    if 'train_dataset' in globals(): del globals()['train_dataset']
    if 'test_dataset' in globals(): del globals()['test_dataset']
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
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

    if args.repos_private:
        assert os.environ.get('HF_TOKEN') is not None, "Set up HF_TOKEN if the model_id repo is private"
        HF_TOKEN = os.environ.get('HF_TOKEN')
        login(token=HF_TOKEN)

    if args.push_to_hub or args.create_new_repo:
        assert os.environ.get('HF_TOKEN') is not None, "Set up HF_TOKEN if you want to push model to a HF repo"
        assert args.push_to_hub_repo_id is not None, "push_to_hub_repo_id is not provided"
        HF_TOKEN = os.environ.get('HF_TOKEN')
        login(token=HF_TOKEN)

    if args.save_local:
        assert args.output_dir is not None, "output_dir is not provided"

    if args.report_to=='wandb':
        assert os.environ.get('WANDB_API_KEY') is not None, "Set up WANDB_API_KEY in order to report to 'wandb'"
        WANDB_API_KEY = os.environ.get('WANDB_API_KEY')
        wandb.login(key=WANDB_API_KEY)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, processor = get_model_and_processor(args, device)

    ### get dataset
    train_dataset, test_dataset = get_dataset()
    train_dataset = [create_conversation_template_for_training(example) for example in train_dataset]
    test_dataset = [create_conversation_template_for_training(example) for example in test_dataset]

    ### pretrain the randomly initialized audio_projector weights
    pretrain_audio_projector(model, processor, train_dataset, test_dataset, args)

    # evaluate pretrained model using Word Error Rate (WER) metric
    wer = evaluate_model_wer(model, processor, device, eval_dataset=test_dataset,
                             eval_batch_size=args.per_device_eval_batch_size, max_new_tokens=args.max_new_tokens)
    logger.info(f"WER score, over eval_dataset:   {wer:.2f} (%)")

    # Save model in local directory
    if args.save_local:
        model.save_pretrained(args.output_dir)
        logger.info(f"model with pretrained audio projector module saved to local directory.")

    ### create a repository
    if args.create_new_repo:
        api = HfApi()
        api.create_repo(repo_id=args.push_to_hub_repo_id, private=True)
        logger.info(f"New repo created for pushing model with pretrained audio projector module, repo_id: {args.push_to_hub_repo_id}")

    # push processor and model to repo
    if args.push_to_hub:
        processor.push_to_hub(args.push_to_hub_repo_id)
        model.push_to_hub(args.push_to_hub_repo_id)
        logger.info(f"model with pretrained audio projector module and processor pushed to repo_id: {args.push_to_hub_repo_id}")

    clear_memory()