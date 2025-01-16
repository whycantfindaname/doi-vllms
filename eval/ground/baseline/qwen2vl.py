from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
import re
from tqdm import tqdm

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "../models/qwen2-vl-7b-instruct",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
# default processer
processor = AutoProcessor.from_pretrained("../models/qwen2-vl-7b-instruct", max_pixels=4096*28*28)

def generate_response(messages):
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return output_text
    
# ['<|object_ref_start|>edge ringing effect affecting the edge of the table near the center bottom.<|object_ref_end|><|box_start|>(1,601),(996,996)<|box_end|><|im_end|>']
# 取出其中的bbox坐标，以[x1, y1, x2, y2]的形式返回
def get_bbox_coordinates(response):
    # 修正正则表达式
    match = re.search(r'<\|box_start\|\>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>', response[0])
    if match:
        # 提取坐标并转换为整数列表
        return [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
    else:
        # 返回空列表表示未找到 bbox
        return response[0]


meta_file = 'data/meta_json/benchmark-v1/release/dist_info_v1.json'
image_folder = '../datasets/images/doi-images-all'
save_file = 'results/doi_bench/grounding/baseline/qwen2vl.json'
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

    dist = meta_item['distortions']
    if isinstance(dist, str):
        continue
    gen_dist = []
    for distortion in dist:
        dist_name = distortion['distortion']
        dist_pos = distortion['position']
        dist_info = f"{dist_name.lower()} affecting {dist_pos.lower()}"
        dist_id = distortion['id']
        question = f"Please provide the bounding box coordinates of the region this sentence describes: <|object_ref_start|>{dist_info}<|object_ref_end|>"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        output_text = generate_response(messages)
        bbox = get_bbox_coordinates(output_text)
        gen_dist.append({'id': dist_id, 'description': dist_info, 'bbox': bbox})
    new_data.append({'image': image, 'distortions': gen_dist})
    with open(save_file, 'w') as f:
        json.dump(new_data, f, indent=4)
        


# ['text in the upper-left corner(132,191),(496,231)']
# ['out of focus blur in the text in the upper-left corner(126,191),(496,233)']
# ['out of focus blur(234,0),(911,341)']