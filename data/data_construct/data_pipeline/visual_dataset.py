from concurrent.futures import ThreadPoolExecutor
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
import re
torch.manual_seed(1234)

meta_file = 'data/meta_json/benchmark-v1/test/dist_info_v2_clean.json'
meta_file1 = 'data/meta_json/train-v1/test/dist_info_v2_clean.json'
image_folder = '../datasets/images/doi-images-all'
tokenizer = AutoTokenizer.from_pretrained("../models/Qwen-VL-Chat", trust_remote_code=True)

# 加载元数据
with open(meta_file, 'r') as f:
    meta_data1 = json.load(f)
with open(meta_file1, 'r') as f:
    meta_data2 = json.load(f)

def process_single_bbox(meta_item):
    image = meta_item['image']
    img_path = os.path.join(image_folder, image)
    save_folder = os.path.join('../datasets/images/bench_vis_dist', image.split('.')[0])
    os.makedirs(save_folder, exist_ok=True)
    
    # 检查是否已经存在处理结果
    if os.path.exists(save_folder) and os.listdir(save_folder):
        return
    
    dist = meta_item['distortions']
    if isinstance(dist, str):
        return
    
    i = 1
    query = tokenizer.from_list_format([
        {'image': img_path},
    ])
    
    for dist_item in dist:
        save_name = os.path.join(save_folder, f'{i}.jpg')
        coord = dist_item['coordinates']
        response = f"<box>({coord[0], coord[1]}),({coord[2], coord[3]})</box>"
        history = [(query, response)]
        image = tokenizer.draw_bbox_on_latest_picture(response, history)
        if image:
            image.save(save_name)
            i += 1

    assert i == len(dist) + 1, "number of generated images not equal to number of distortions"


def process_all_bbox(meta_item):
    image = meta_item['image']
    img_path = os.path.join(image_folder, image)
    # if image != 'LIVEfb_VOC2012__2009_004232.jpg':
    #     return
    save_folder_base = '../datasets/images/train_vis_dist'
    save_folder_base = '../datasets/images/bench_vis_dist_v2'
    # save_folder_base = 'test'
    os.makedirs(save_folder_base, exist_ok=True)
    
    dist = meta_item['distortions']
    if isinstance(dist, str):
        return
    
    # Group distortions by type
    grouped_distortions = {}
    for dist_item in dist:
        distortion_type = dist_item['distortion']
        if distortion_type not in grouped_distortions:
            grouped_distortions[distortion_type] = []
        grouped_distortions[distortion_type].append(dist_item)
    
    # Visualize each distortion type on a single image
    for distortion_type, items in grouped_distortions.items():
        i = 1
        query = tokenizer.from_list_format([
            {'image': img_path},
        ])
        response = ""
        
        # Create a folder for each image
        image_folder_path = os.path.join(save_folder_base, os.path.splitext(image)[0])
        os.makedirs(image_folder_path, exist_ok=True)
        distortion_type = distortion_type.replace(" ", "_")      # 替换文件名中的空格
        save_path = os.path.join(image_folder_path, f"{distortion_type}.jpg")
        
        # Skip if the result already exists
        if os.path.exists(save_path):
            continue
        
        for dist_item in items:
            coord = dist_item['coordinates']
            id_str = dist_item['id']
            number = re.search(r'\d+', id_str)
            if number:
                id = number.group()
            response += f"<ref>{id}</ref><box>({coord[0]}, {coord[1]}), ({coord[2]}, {coord[3]})</box>"
            i += 1
        
        history = [(query, response)]
        img = tokenizer.draw_bbox_on_latest_picture(response, history)
        if img:
            img.save(save_path)
        
        if i != len(items) + 1:
            print(f"Number of generated bounding boxes {i-1} does not match the distortions {len(items)}")
            print(items)
            input()
# 使用多线程处理
with ThreadPoolExecutor(max_workers=8) as executor:
    list(tqdm(executor.map(process_all_bbox, meta_data1), total=len(meta_data1)))