from PIL import Image
import base64
import os
import json
from collections import defaultdict
from typing import Optional
import re
def encode_img(img_path):
    ext = os.path.splitext(img_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif ext == ".png":
        mime_type = "image/png"
    elif ext == ".webp":
        mime_type = "image/webp"
    elif ext == ".bmp":
        # 转换成jpg格式后编码
        img = Image.open(img_path)
        img.save(img_path.replace(ext, ".jpg"), "JPEG")
        mime_type = "image/jpeg"
        img_path = img_path.replace(ext, ".jpg")
    else:
        raise ValueError("Unsupported image format")
    print(img_path)
    with open(img_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    return mime_type, img_base64


def convert_train_dist_info(dist_class):
    formatted_bboxes = []
    bbox_counter = 1
    
    # Iterate over the local distortions
    for distortion, bboxes in dist_class.items():
        for bbox in bboxes:
            formatted_bbox = (
                f"bbox {bbox_counter}: {{\"distortion\": {distortion}, \"coordinates\":{{'tl':{{\"x\":{bbox['tl']['x']}, \"y\":{bbox['tl']['y']}}}, "
                f"'br':{{\"x\":{bbox['br']['x']}, \"y\":{bbox['br']['y']}}}}}}}"
            )
            formatted_bboxes.append(formatted_bbox)
            bbox_counter += 1
    
    return "\n".join(formatted_bboxes)

def load_json(file_path):
    """Load a JSON file and return its contents."""
    with open(file_path, "r") as file:
        return json.load(file) 
    
def conver_meta_data_to_gpt(meta_data: dict, description: Optional[str] = None):
    text = ''
    level = meta_data['level']
    mos = meta_data['mos']
    if description is None:
        text += f'[Image Infomation]\nMos: {mos}, Level: {level}\n'
    else:
        text += f'[Image Infomation]\nMos: {mos}, Level: {level}, Description: {description}\n'
    distortions = meta_data['distortions']
    text += f'[Distortion Infomation]\n'
    for distortion in distortions:
        id = distortion['id']
        dist = distortion['distortion']
        position = distortion['position']
        severity = distortion['severity']
        visual_manifestation = distortion['visual manifestation']
        perception_impact = distortion['perception impact']
        region_quality = distortion['region_quality_scores']
        region_importance = distortion['region_importance_score']
        text += f'<{id}> Distortion: {dist}, Position: {position}, Severity: {severity}, Visual Manifestation: {visual_manifestation}, Perception Impact: {perception_impact}, Region Quality: {region_quality}, Region Importance: {region_importance}\n'
    return text

def clean_aigc_data(data):
    print("Before cleaning:", len(data))
    for item in data:
        if "AIGC" in item['image']:
            data.remove(item)
    print("After cleaning:", len(data))
    return data

def clean_text(text):
    # Remove any tags such as <bbox 1> </bbox 1> or <1> </1>
    cleaned_text = re.sub(r'<[^>]+>', '', text)
    return cleaned_text

def process_images(images):
    # 用来统计相同键出现的图片数量
    key_count = defaultdict(int)

    # 遍历所有图片
    for image in images:
        global_keys = set(image["global"].keys())  # 获取 global 中的所有键
        local_keys = set(image["local"].keys())    # 获取 local 中的所有键
        
        # 查找 global 和 local 中共同的键
        common_keys = global_keys & local_keys  # 交集操作
        
        # 如果有共同的键，记录该键的出现次数并删除 global 中的该键
        for key in common_keys:
            key_count[key] += 1
            del image["global"][key]
        

    # 返回统计结果
    return key_count, images

def show_number(data):
    distortion_count = defaultdict(int)
    for item in data:
        if item["global"] != {}:
            for issue, details in list(item["global"].items()):  # Use list to avoid modifying dict while iterating
                distortion_count[issue] += len(details)
        if item["local"] != {}:
            for issue, details in list(item["local"].items()):  # Use list to avoid modifying dict while iterating
                distortion_count[issue] += len(details)
    print(distortion_count)
    
def show_processed_number(data):
    distortion_count = defaultdict(int)
    for item in data:
        if isinstance(item['distortions'], list):
            for distortion in item['distortions']:
                distortion_count[distortion['distortion']] += 1
        else:
            continue
    print(distortion_count)
def check_repeat_images(data):
    # 提取所有图像路径
    image_paths = [item['image'] for item in data]
    # 统计每个图像出现的次数
    image_counts = {}
    for image in image_paths:
        image_counts[image] = image_counts.get(image, 0) + 1
    
    # 检查是否有重复图像
    repeat_images = [image for image, count in image_counts.items() if count > 1]
    if repeat_images:
        print("Warning: There are repeat images in the data.")
    
    # 输出重复图像的数量
    print("Number of repeat images:", len(repeat_images))
    print("Repeat images:", repeat_images[0:10])

    # 删除重复图像并保留第一个出现的条目
    unique_data = []
    seen_images = set()
    for item in data:
        if item['image'] not in seen_images:
            unique_data.append(item)
            seen_images.add(item['image'])
    
    return unique_data

def check_sanity(data):
    for item in data:
        mos = item['mos']
        bboxes = {**item['global'], **item['local']}
        if mos < 4 and len(bboxes) == 0:
            print(item)
            data.remove(item)
    print("number of images after clean:", len(data))
    return data

        
def clean_dist(data):
    # Iterate over each item
    for item in data:
        if item["global"] != {}:
            for issue, details in list(item["global"].items()):  # Use list to avoid modifying dict while iterating
                if len(details) == 0:
                    item['global'].pop(issue, None)
                # If the issue is in the specified list, remove it
                elif issue in ["Interlaced scanning", "Banding effect", "Moiré pattern"]:
                    item['global'].pop(issue, None)
                # If the issue is "Excessive darkness", change it to "Underexposure"
                elif issue == "Excessive darkness":
                    item['global']["Underexposure"] = item['global'].pop(issue)
                elif issue == "Low clarity":
                    for issue_detail in details:
                        try:
                            if "unimportant" in issue_detail['seagull']['importance'] or 'minor' in issue_detail['seagull']['importance'] or "excellent" in issue_detail['seagull']['quality'] or "good" in issue_detail['seagull']['quality']:
                                # print(f"  Removing {issue_detail} from {issue}")
                                # input()  # To inspect the issue before removing it (optional)
                                item['global'][issue].remove(issue_detail)
                        except KeyError:
                            print(f"  No seagull info found in {issue_detail}")

                else:
                    continue
        if item["local"] != {}:
            for issue, details in list(item["local"].items()):  # Use list to avoid modifying dict while iterating
                if len(details) == 0:
                    item['local'].pop(issue, None)
                # If the issue is in the specified list, remove it
                elif issue in ["Interlaced scanning", "Banding effect", "Moiré pattern"]:
                    item['local'].pop(issue, None)
                # If the issue is "Excessive darkness", change it to "Underexposure"
                elif issue == "Excessive darkness":
                    item['local']["Underexposure"] = item['local'].pop(issue)
                elif issue == "Low clarity":
                    for issue_detail in details:
                        try:
                            if "unimportant" in issue_detail['seagull']['importance'] or 'minor' in issue_detail['seagull']['importance'] or "excellent" in issue_detail['seagull']['quality'] or "good" in issue_detail['seagull']['quality']:
                                # print(f"  Removing {issue_detail} from {issue}")
                                # input()  # To inspect the issue before removing it (optional)
                                item['local'][issue].remove(issue_detail)
                        except KeyError:
                            print(f"  No seagull info found in {issue_detail}")
                else:
                    continue
                    
    return data

def combine_seagull_importance_quality(importance_file, quality_file):
    data = []
    
    # Load both JSON files
    with open(importance_file, 'r') as f:
        importance_data = json.load(f)
        
    with open(quality_file, 'r') as f:
        quality_data = json.load(f)
    
    # Iterate over the data
    for importance_item, quality_item in zip(importance_data, quality_data):
        if importance_item['image'] == quality_item['image']:
            # Merge global issues
            if importance_item['global'] != {}:
                for issue, details in importance_item['global'].items():
                    for i in range(len(details)):
                        if 'seagull' in details[i]:
                            # Try merging seagull quality data
                            try: 
                                quality_seagull = quality_item['global'][issue][i]['seagull']['quality']
                                details[i]['seagull']['quality'] = quality_seagull
                            except KeyError:
                                print(f"  No seagull info found in quality data for global issue '{issue}' at index {i}")
                        else:
                            print(f"  No seagull info found in importance data for global issue '{issue}' at index {i}")
            else:
                print("  No global issues found in importance data")
        
            # Merge local issues
            if importance_item['local'] != {}:
                for issue, details in importance_item['local'].items():
                    if issue in quality_item['local']:
                        for i in range(len(details)):    
                            if 'seagull' in details[i]:
                                # Try merging seagull quality data for local issues
                                try: 
                                    quality_seagull = quality_item['local'][issue][i]['seagull']['quality']
                                    details[i]['seagull']['quality'] = quality_seagull
                                except KeyError:
                                    print(f"  No seagull info found in quality data for local issue '{issue}' at index {i}") 
                            else:
                                print(f"  No seagull info found in importance data for local issue '{issue}' at index {i}")
                    else:
                        print(f"  Local issue '{issue}' not found in quality data")
            else:
                print("  No local issues found in importance data") 
            
            # Add the combined item to the data
            data.append(importance_item)
        else:
            print(f"Skipping image '{importance_item['image']}' as it doesn't match quality data.")
    
    return data


# def combine_gpt_dist_with_seagull(importance_file, quality_file):
#     data = []
#     with open(importance_file, 'r') as f:
#         importance_data = json.load(f)
#     with open(quality_file, 'r') as f:
#         quality_data = json.load(f)
#     for quality_item, importance_item in zip(quality_data, importance_data):
#         if quality_item['image'] == importance_item['image']:
            
            
if __name__ == '__main__':
    meta_data = load_json('data/meta_json/benchmark/test/test_dist_info_v1.json')[0]
    text = conver_meta_data_to_gpt(meta_data)
    print(repr(text))