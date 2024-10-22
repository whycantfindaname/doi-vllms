import os
import warnings
from io import BytesIO

import debugpy
import requests
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlavaForConditionalGeneration,
)
from transformers.models.clip.image_processing_clip import CLIPImageProcessor


def setup_debugger(port=9501):
    try:
        debugpy.listen(("localhost", port))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    except Exception as e:
        print(f"Debugger setup failed: {e}")


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_pretrained_model(
    model_path,
    model_base=None,
    model_name=None,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
):
    kwargs = {"device_map": {"": device} if device != "cuda" else device_map}

    if load_8bit or load_4bit:
        kwargs["load_in_8bit"] = load_8bit
        if load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
    else:
        kwargs["torch_dtype"] = torch.float16
    if model_name is None:
        model_name = os.path.basename(model_path)
    if "llava-interleave" in model_name.lower():
        processor, tokenizer, image_processor = None, None, None
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "LoRA model specified but no `model_base` provided. Provide `model_base` when loading a LoRA model."
            )
        if "lora" in model_name.lower() and model_base:
            processor = AutoProcessor.from_pretrained(model_base, use_fast=False)
            tokenizer, image_processor = processor.tokenizer, processor.image_processor
            print(f"Loading finetuned llava-interleave model {model_name}...")
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )
        elif model_base:
            print("Loading llava-interleave from base model...")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            config = AutoConfig.from_pretrained(model_path)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=config, **kwargs
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )

        context_len = getattr(model.config, "max_sequence_length", 2048)
        return (
            processor,
            model,
            context_len
            if processor
            else (tokenizer, model, image_processor, context_len),
        )

    elif "qwen-vl-chat" in model_name.lower():
        # Load Qwen-VL-Chat model
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "LoRA model specified but no `model_base` provided. Provide `model_base` when loading a LoRA model."
            )
        elif "lora" in model_name.lower() and model_base:
            tokenizer = AutoTokenizer.from_pretrained(
                model_base, trust_remote_code=True, use_fast=False
            )
            print(f"Loading finetuned qwen-vl-chat model {model_name}...")
            model = AutoModelForCausalLM.from_pretrained(
                
                model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
            )
            print("Convert to FP16...")
            model.to(torch.float16)
        elif "full" in model_name.lower():
            print("Loading full finetuned qwen-vl-chat...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                model_base, trust_remote_code=True, use_fast=False
                )
            except:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True, use_fast=False
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
            )
            print("Convert to FP16...")
            model.to(torch.float16)
        elif model_base:
            print("Loading qwen-vl-chat from base model...")
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            config = AutoConfig.from_pretrained(model_base)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=config, **kwargs
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )
        context_len = getattr(model.config, "max_sequence_length", 2048)
        return tokenizer, model, context_len

    else:
        # Load language model
        if model_base is not None:
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, **kwargs
            )
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )

    image_processor = CLIPImageProcessor.from_pretrained(model_path)
    context_len = getattr(model.config, "max_sequence_length", 2048)

    return tokenizer, model, image_processor, context_len


def get_llava_answer(model, processor, device, image, conversation, **kwargs):
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(prompt, image, return_tensors="pt").to(torch.float16).to(device)
    output = model.generate(**inputs, **kwargs)
    answer = processor.decode(output[0][2:], skip_special_tokens=True).split("\n")[-1]
    return answer


def add_history_chat(chat, question, prev_answer=None, image=None):
    if prev_answer:
        answer_temp = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": prev_answer},
            ],
        }
        chat["conversation"].append(answer_temp)
    if image:
        question_temp = {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
            ],
        }
    else:
        question_temp = {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
    if question:
        chat["conversation"].append(question_temp)
    return chat
