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
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
    Qwen2VLForConditionalGeneration
)
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import math
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
    torch_dtype=torch.bfloat16,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_custom_processor=True
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
        kwargs["torch_dtype"] = torch_dtype
    if model_name is None:
        model_name = os.path.basename(model_path)
    if "llava-onevision" in model_name.lower():
        # Load llava-onevision model
        processor, tokenizer = None, None
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "LoRA model specified but no `model_base` provided. Provide `model_base` when loading a LoRA model."
            )
        if "lora" in model_name.lower() and model_base:
            try:
                processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
            except:
                processor = AutoProcessor.from_pretrained(model_base, use_fast=False)
            try:    
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            except:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            try:
                config = AutoConfig.from_pretrained(model_path)
            except:
                config = AutoConfig.from_pretrained(model_base)
            print(f"Loading lora finetuned llava-onevision model {model_name} from {model_path}...")
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )
            print(f"Load model in {model.dtype}")
        elif model_base:
            print(f"Loading original llava-onevision from base model {model_base}...")
            processor = AutoProcessor.from_pretrained(model_base, use_fast=False)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            config = AutoConfig.from_pretrained(model_base)
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=config, **kwargs
            )
            print(f"Load model in {model.dtype}")
        else:
            print(f"Loading model from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            processor  = AutoProcessor.from_pretrained(model_path, use_fast=False)
            config = AutoConfig.from_pretrained(model_path)
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )
            print(f"Load model in {model.dtype}")

        return model, processor, tokenizer, config

    elif "qwen-vl-chat" in model_name.lower():
        # Load Qwen-VL-Chat model
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "LoRA model specified but no `model_base` provided. Provide `model_base` when loading a LoRA model."
            )
        elif "lora" in model_name.lower() and model_base:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True, use_fast=False
                )
            except:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_base, trust_remote_code=True, use_fast=False
                )
            try:
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            except:
                config = AutoConfig.from_pretrained(model_base, trust_remote_code=True)
            print(f"Loading lora finetuned qwen-vl-chat model {model_name} from {model_path}...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
            )
            print(f"Load model in {model.dtype}")
        elif "full" in model_name.lower():
            print("Loading full finetuned qwen-vl-chat...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
                )
            except:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_base, trust_remote_code=True, use_fast=False
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
            )
            print(f"Load model in {model.dtype}")
        elif model_base:
            print(f"Loading qwen-vl-chat from base model {model_base}...")
            tokenizer = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True, use_fast=False)
            config = AutoConfig.from_pretrained(model_base, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=config, trust_remote_code=True, **kwargs
            )
            print(f"Load model in {model.dtype}")
        else:
            print(f"Loading model from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs
            )
            print(f"Load model in {model.dtype}")
        
        return model, None, tokenizer, config

    elif "qwen2-vl" in model_name.lower():
        # Load Qwen2-VL model
        processor, tokenizer, config = None, None, None
        if "lora" in model_name.lower() and model_base is None:
            warnings.warn(
                "LoRA model specified but no `model_base` provided. Provide `model_base` when loading a LoRA model."
            )
        if "lora" in model_name.lower() and model_base:
            print(f"Loading lora finetuned qwen2-vl model {model_name} from {model_path}...")
            if use_custom_processor:
                processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
            else:
                processor = AutoProcessor.from_pretrained(model_base, use_fast=False)
            try:    
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            except:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            try:
                config = AutoConfig.from_pretrained(model_path)
            except:
                config = AutoConfig.from_pretrained(model_base)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )
            print(f"Load model in {model.dtype}")
        elif model_base:
            print(f"Loading original qwen2-vl from base model {model_base}...")
            processor = AutoProcessor.from_pretrained(model_base, use_fast=False, max_pixels=4096*28*28)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            config = AutoConfig.from_pretrained(model_base)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_base, low_cpu_mem_usage=True, config=config, **kwargs
            )
            print(f"Load model in {model.dtype}")
            print(f"Load model to {model.hf_device_map}")
        else:
            print(f"Loading model from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            processor  = AutoProcessor.from_pretrained(model_path, use_fast=False)
            config = AutoConfig.from_pretrained(model_path)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, low_cpu_mem_usage=True, **kwargs
            )
            print(f"Load model in {model.dtype}")

        return model, processor, tokenizer, config
    
    elif "internvl2" in model_name.lower():
        # Load InternVL2 model
        print(f"Loading internvl2 model from {model_path}...")
        processor, tokenizer, config = None, None, None
        device_map = split_model('InternVL2-8B')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            device_map=device_map).eval()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        print(f"Load model in {model.dtype}")
        print(f"Load model to {model.hf_device_map}")
        return model, processor, tokenizer, config
        
        

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

# internvl2 processor
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def internvl2_load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def tokenizer_image_token(prompt, tokenizer, image_token_index=-200, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids if len(chunk) > 0 else [] for chunk in prompt.split("<|image|>")]
    # print(prompt_chunks)

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


if __name__ == "__main__":
    # # load original model
    # model_path = "../models/qwen2-vl-7b-instruct"
    # model_path = '../models/llava-onevision-qwen2-7b-ov-hf'
    # # model_path = "../models/Qwen-VL-Chat"
    # model, processor, tokenizer, config = load_pretrained_model(model_path)

    # load lora finetuned model
    model_path = "checkpoints/qwen2-vl-7b-instruct_lora-True_qlora-False-gvlmiqa-v0.2-score"
    model_base = "../models/qwen2-vl-7b-instruct"
    model, processor, tokenizer, config = load_pretrained_model(model_path, model_base=model_base, device="cuda:1")
    
    