from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import json
import os
import re
from tqdm import tqdm

model = model = AutoModelForCausalLM.from_pretrained(
    "../models/Qwen-VL-Chat",
    fp16=True,
    trust_remote_code=True,
    device_map="auto"
).eval()
# default processer
tokenizer = AutoTokenizer.from_pretrained("../models/Qwen-VL-Chat", trust_remote_code=True)

def generate_response(messages):
    # Preparation for inference
    query = tokenizer.from_list_format(messages)
    response, history = model.chat(tokenizer, query=query, history=None)
    return response
    
# <ref>out of focus blur affecting the text in the upper-left corner.</ref><box>(141,187),(538,239)</box>
# 取出其中的bbox坐标，以[x1, y1, x2, y2]的形式返回
def get_bbox_coordinates(response):
    # 修正正则表达式，确保匹配 "<box>(x1,y1),(x2,y2)</box>"
    match = re.search(r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>', response)
    if match:
        # 提取坐标并转换为整数列表
        return [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
    else:
        # 如果未匹配，返回原始输入
        return response


meta_file = 'data/meta_json/benchmark-v1/release/dist_info_v1.json'
image_folder = '../datasets/images/doi-images-all'
save_file = 'results/doi_bench/grounding/baseline/qwenvl.json'
os.makedirs(os.path.dirname(save_file), exist_ok=True)

with open(meta_file, 'r') as f:
    meta_data = json.load(f)

new_data = []

for meta_item in tqdm(meta_data, total=len(meta_data)):
    image = meta_item['image']
    img_path = os.path.join(image_folder, image)

    dist = meta_item['distortions']
    if isinstance(dist, str):
        continue
    gen_dist = []
    for distortion in dist:
        dist_name = distortion['distortion']
        dist_pos = distortion['position']
        dist_info = f"{dist_name.lower()} affecting {dist_pos.lower()}"
        dist_id = distortion['id']
        question = f"Please provide the bounding box coordinates of the region this sentence describes: <ref>{dist_info}</ref>"
        messages = [
            {"image": img_path},
            {"text": question}
        ]
        output_text = generate_response(messages)
        bbox = get_bbox_coordinates(output_text)
        gen_dist.append({'id': dist_id, 'description': dist_info, 'bbox': bbox})
    new_data.append({'image': image, 'distortions': gen_dist})
    print(new_data)

    with open(save_file, 'w') as f:
        json.dump(new_data, f, indent=4)
        


# ['text in the upper-left corner(132,191),(496,231)']
# ['out of focus blur in the text in the upper-left corner(126,191),(496,233)']
# ['out of focus blur(234,0),(911,341)']