import os
import json

meta_file = 'data/meta_json/assessment_final_484_主观验证无误.json'
bench_file = 'data/meta_json/benchmark/release/dist_info_v1.json'
train_file = 'data/meta_json/train/release/assess_v1.json'
new_train_file = 'data/meta_json/train-v1/release/dist_info_v1.json'
new_bench_file = 'data/meta_json/benchmark-v1/release/dist_info_v1.json'

# 创建所需目录
os.makedirs(os.path.dirname(new_train_file), exist_ok=True)
os.makedirs(os.path.dirname(new_bench_file), exist_ok=True)

# 读取数据
with open(meta_file, 'r') as f:
    meta_data = json.load(f)
print(len(meta_data))

with open(bench_file, 'r') as f:
    bench_data = json.load(f)
print(len(bench_data))

with open(train_file, 'r') as f:
    train_data = json.load(f)
print(len(train_data))

# 初始化
meta_filenames = {item['filename'] for item in meta_data}
total_data = bench_data + train_data

new_bench_data = []
new_train_data = []
seen_images = set()

# 分类数据
for total_item in total_data:
    image = total_item['image']
    if image in seen_images:
        continue  # 跳过重复图像
    seen_images.add(image)
    if image in meta_filenames:
        new_bench_data.append(total_item)
    else:
        new_train_data.append(total_item)

print(len(new_bench_data))
print(len(new_train_data))
input()
# 保存新数据
with open(new_bench_file, 'w') as f:
    json.dump(new_bench_data, f, indent=4, ensure_ascii=False)

with open(new_train_file, 'w') as f:
    json.dump(new_train_data, f, indent=4, ensure_ascii=False)
