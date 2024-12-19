from collections import defaultdict
import json

file = '../datasets/val_json/benchmark_2k_normalized.json'
save_path = '../datasets/doi_json/benchmark_2k_seagull_cleaned.json'
with open(save_path, 'r') as f:
    images = json.load(f)

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

key_count, new_images = process_images(images)
key_count1, _ = process_images(new_images)
# 打印统计结果
for key, count in key_count.items():
    print(f"Key: {key}, Count: {count}")
    
for key, count in key_count1.items():   
    print(f"Key: {key}, Count: {count}")

# # 保存结果
# with open(save_path, 'w') as f:
#     json.dump(new_images, f, indent=4, ensure_ascii=False)
