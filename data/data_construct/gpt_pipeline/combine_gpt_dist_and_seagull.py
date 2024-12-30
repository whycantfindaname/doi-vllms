import json
import os
gpt_dist_path = 'data/meta_json/train/test/test_dist_info_v1.json'
seagull_path = 'data/meta_json/train/release/train_10k_seagull_clean.json'
save_path = 'data/meta_json/train/test/input_dist_info_v1.json'
gpt_dist_path = 'data/meta_json/benchmark/test/test_dist_info_v1.json'
seagull_path = 'data/meta_json/benchmark/release/benchmark_2k_seagull_clean.json'
save_path = 'data/meta_json/benchmark/test/input_dist_info_v1.json'

with open(gpt_dist_path, 'r') as f:
    gpt_dist_data = json.load(f)
print(f"Length of gpt_dist_data: {len(gpt_dist_data)}")

with open(seagull_path, 'r') as f:
    seagull_data = json.load(f)

def check_repeat_images(data):
    images = set(data['image'] for data in data)
    if len(images)!= len(data):
        print("Warning: there are repeat images in data.")

unmatched_images = []
for gpt_dist_item in gpt_dist_data:
    image = gpt_dist_item['image']
    try:
        seagull_item = next(item for item in seagull_data if item['image'] == image)
    except StopIteration:
        unmatched_images.append(image)
    
print(f"Length of seagull_data: {len(seagull_data)}")
print(f"Found {len(unmatched_images)} unmatched images in seagull_data.")
print(f"Some unmatched_gpt_images: {unmatched_images[:10]}")

# check_repeat_images(gpt_dist_data)
input()
new_data = []
processed_images = set()  # 用来记录已经处理过的图像，避免重复
skipped_images = []  # 记录跳过的图像
unmatched_gpt_images = []  # 记录没有匹配的 gpt_dist_item
duplicates = []  # 记录重复的 seagull images

for seagull_item in seagull_data:
    image = seagull_item['image']
    
    # 检查重复的图片
    if image in processed_images:
        duplicates.append(image)
        continue
    processed_images.add(image)
    
    try:
        # 尝试从 gpt_dist_data 中找到匹配的条目
        gpt_dist_item = next(item for item in gpt_dist_data if item['image'] == image)
    except StopIteration:
        skipped_images.append(f"{image} not found in gpt_dist_data")
        continue
    
    if isinstance(gpt_dist_item['distortions'], list):
        meta_bboxes = gpt_dist_item['distortions'].copy()
        global_dist = seagull_item['global']
        local_dist = seagull_item['local']
        dist_class = {**global_dist, **local_dist}
        bbox_count = 0
        for distortion, bboxes in dist_class.items():
            for bbox in bboxes:
                coordinates = [
                    bbox["tl"]["x"],
                    bbox["tl"]["y"],
                    bbox["br"]["x"],
                    bbox["br"]["y"]
                ]
                try:
                    region_quality_score = bbox["seagull"]["quality"]
                    region_importance_score = bbox["seagull"]["importance"]
                except KeyError:
                    skipped_images.append(f"No seagull data for {image}, bbox {bbox}")
                    continue
                meta_bboxes[bbox_count]['distortion'] = distortion
                meta_bboxes[bbox_count]['coordinates'] = coordinates
                meta_bboxes[bbox_count]['region_quality_scores'] = region_quality_score
                meta_bboxes[bbox_count]['region_importance_score'] = region_importance_score
                bbox_count += 1
        if bbox_count != len(meta_bboxes):
            skipped_images.append(f"Error processing {image}: bbox count mismatch")
            continue
    else:
        meta_bboxes = gpt_dist_item['distortions']

    mos = seagull_item['mos']
    level = seagull_item['level']
    width = seagull_item['width']
    height = seagull_item['height']
    new_data.append({
        'image': image,
        'mos': mos,
        'level': level,
        'width': width,
        'height': height,
        'distortions': meta_bboxes
    })

# 打印调试信息
print(f"Length of new_data: {len(new_data)}")
print(f"Found {len(skipped_images)} skipped images.")
print(f"Some skipped images: {skipped_images[:10]}")  # 打印前10个跳过的图像
print(f"Found {len(unmatched_gpt_images)} unmatched gpt_dist_data images.")
print(f"Some unmatched gpt_dist_data images: {unmatched_gpt_images[:10]}")
print(f"Found {len(duplicates)} duplicate images.")
print(f"Some duplicate images: {duplicates[:10]}")

# 保存 new_data
with open(save_path, 'w') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)

print(f"Length of gpt_dist_data after processing: {len(gpt_dist_data)}")
with open(gpt_dist_path, 'w') as f:
    json.dump(gpt_dist_data, f, indent=4, ensure_ascii=False)