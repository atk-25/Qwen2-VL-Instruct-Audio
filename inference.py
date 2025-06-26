import os
import argparse
import logging
import gc
import time
import torch
from huggingface_hub import login
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_audio_info
from transformers.utils import is_flash_attn_2_available
from peft import PeftModel
from datasets import load_dataset
from evaluate import load


logger = logging.getLogger(__name__)


def arg_parser():

    parser = argparse.ArgumentParser()

    attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

    parser.add_argument("--base_model_id", type=str, default=None)
    parser.add_argument("--lora_adapters_id", type=str, default=None)
    parser.add_argument("--use_quantized_model", action="store_true")
    parser.add_argument("--qlora_model_id", type=str, default=None)
    parser.add_argument("--repos_private", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default=attn_implementation)
    parser.add_argument("--per_device_inference_batch_size", type=int, default=8)

    parser.add_argument("--max_new_tokens", type=int, default=256)

    args = parser.parse_args()

    return args


system_message = """You are a Vision Language and Audio Model specialized in interpreting visual and audio data.
Your task is to transcribe the provided audio signal.
Focus on delivering accurate, succinct answers based on the audio information. Avoid additional explanation unless absolutely necessary."""

def create_conversation_template_for_inference(audio):

    prompt = "Transcribe this audio."

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
    ]

    return messages


def get_model_and_processor(args, device):

    # download Qwen2-VL model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.base_model_id, torch_dtype=torch.bfloat16, device_map=device, attn_implementation=args.attn_implementation
    )
    processor = Qwen2VLProcessor.from_pretrained(args.base_model_id)

    model = PeftModel.from_pretrained(model, args.lora_adapters_id)
    # merge the (quantized) base model and adapter for inference
    model = model.merge_and_unload()

    return model, processor


def get_model_and_processor_qlora(args, device):

    if args.qlora_model_id is not None:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.qlora_model_id, torch_dtype=torch.bfloat16, device_map=device, attn_implementation=args.attn_implementation
        )
        processor = Qwen2VLProcessor.from_pretrained(args.qlora_model_id)
        return model, processor

    else:
        bnb_nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # load an NF4 model
        model_q_nf4 = Qwen2VLForConditionalGeneration.from_pretrained(
            args.base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=bnb_nf4_config,
            attn_implementation=args.attn_implementation,
        )
        processor = Qwen2VLProcessor.from_pretrained(args.base_model_id)

        model = PeftModel.from_pretrained(model_q_nf4, args.lora_adapters_id)
        # merge the (quantized) base model and adapter for inference
        model = model.merge_and_unload()

        return model, processor


def run_inference_asr(model, processor, device, examples, inference_batch_size=8, max_new_tokens=256):
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

        # Get the texts and audios
        if isinstance(examples_batch[0], dict):
            examples_batch = [examples_batch]
        texts = [processor.apply_chat_template(example, tokenize=False, add_generation_prompt=True) for example in
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

        output_texts.extend(output_texts_batch)
        start_idx = end_idx

    return output_texts


def clear_memory():
    if 'train_dataset' in globals(): del globals()['train_dataset']
    if 'test_dataset' in globals(): del globals()['test_dataset']
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'model_q_nf4' in globals(): del globals()['model_q_nf4']
    if 'processor' in globals(): del globals()['processor']
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
        assert os.environ.get('HF_TOKEN') is not None, "Set up HF_TOKEN if the repos are private"
        HF_TOKEN = os.environ.get('HF_TOKEN')
        login(token=HF_TOKEN)

    if args.use_quantized_model==False or (args.use_quantized_model==True and args.qlora_model_id is None):
        assert args.base_model_id is not None, "base_model_id is not provided"
        assert args.lora_adapters_id is not None, "lora_adapters_id is not provided"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get model and processor
    if args.use_quantized_model:
        model, processor = get_model_and_processor_qlora(args, device)
    else:
        model, processor = get_model_and_processor(args, device)

    ### get examples for inference
    test_dataset = load_dataset("speechbrain/LargeScaleASR", data_files=["test/test-00000*"], num_proc=12)
    test_dataset = test_dataset["train"]
    test_dataset = test_dataset.select(range(3))
    test_dataset = test_dataset.to_list()

    ### A single audio input could be an BytesIO object, or it could be in any audio format that librosa.load() can process.
    ### If the audio is already decoded into an array of audio time series, then the input should be a dictionary with keys 'array' and 'sampling_rate'
    audios = [example['wav']['bytes'] for example in test_dataset]
    audios_for_inference = [create_conversation_template_for_inference(audio) for audio in audios]
    logger.info(f"number of audios for inference: {len(audios_for_inference)}")

    # run inference
    inferences_asr = run_inference_asr(model, processor, device, examples=audios_for_inference,
                             inference_batch_size=args.per_device_inference_batch_size, max_new_tokens=args.max_new_tokens)

    for i in range(len(audios_for_inference)):
        print(f"Transcripton of audio {i}: ", inferences_asr[i])
        print(f"Reference of audio {i}:    ", test_dataset[i]['text'])

    clear_memory()
