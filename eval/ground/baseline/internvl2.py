import numpy as np
import torch
import torchvision.transforms as T
import math
import json
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import re
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

def load_image(image_file, input_size=448, max_num=12):
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
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
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

    

def get_bbox_coordinates(response):
    # 匹配 <box> 标签中包含的 [[x1, y1, x2, y2]]
    match = re.search(r'<box>\s*\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]\s*</box>', response)
    if match:
        # 提取坐标并转换为整数列表
        return [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
    else:
        # 返回空列表表示未找到 bbox
        return response

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../models/InternVL2_5-8B', help='path to the model')
    parser.add_argument('--mdp', type=int, default=12, help='maximum dynamic patch size')
    args = parser.parse_args()
    path = args.model_path
    device_map = split_model('InternVL2_5-8B')
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map=device_map).eval()

    print(tokenizer.model_max_length)
    generation_config = dict(max_new_tokens=128, do_sample=True)
    

    meta_file = 'data/meta_json/benchmark-v1/release/dist_info_v1.json'
    image_folder = '../datasets/images/doi-images-all'
    save_file = 'results/doi_bench/grounding/baseline/internvl2.json'
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    with open(meta_file, 'r') as f:
        meta_data = json.load(f)

    new_data = []
    
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            new_data = json.load(f)
    complete_images = set([os.path.basename(item['image']) for item in new_data])
    meta_data = [item for item in meta_data if os.path.basename(item['image']) not in complete_images]
    print(len(meta_data))
    input()
    for meta_item in tqdm(meta_data, total=len(meta_data)):
        image = meta_item['image']
        img_path = os.path.join(image_folder, image)
        pixel_values = load_image(img_path, input_size=448, max_num=args.mdp).to(model.device).to(model.dtype)

        dist = meta_item['distortions']
        if isinstance(dist, str):
            continue
        gen_dist = []
        for distortion in dist:
            dist_name = distortion['distortion']
            dist_pos = distortion['position']
            dist_info = f"{dist_name.lower()} affecting {dist_pos.lower()}"
            dist_id = distortion['id']
            question = f"Please provide the bounding box coordinates of the region this sentence describes: <ref>{dist_info}/ref>"
            output_text = model.chat(tokenizer, pixel_values, question, generation_config)
            bbox = get_bbox_coordinates(output_text)
            print(bbox)
            gen_dist.append({'id': dist_id, 'description': dist_info, 'bbox': bbox})
        new_data.append({'image': image, 'distortions': gen_dist})
        with open(save_file, 'w') as f:
            json.dump(new_data, f, indent=4)
        