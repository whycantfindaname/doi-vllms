import json
import os

meta_file1 = '/home/liaowenjie/桌面/多模态大模型/lmms-finetune/data/meta_json/benchmark-v1/process/dist_info_v2.json'
save_file1 = '/home/liaowenjie/桌面/多模态大模型/lmms-finetune/data/meta_json/benchmark-v1/test/dist_info_v2_clean.json'
meta_file2 = '/home/liaowenjie/桌面/多模态大模型/lmms-finetune/data/meta_json/train-v1/process/dist_info_v2.json'
save_file2 = '/home/liaowenjie/桌面/多模态大模型/lmms-finetune/data/meta_json/train-v1/test/dist_info_v2.json'

def extract_dist_meta(meta_file, save_file):
    with open(meta_file, 'r') as f:
        meta_data = json.load(f)
    
    for meta_item in meta_data:
        if isinstance(meta_item['distortions'], str):
            continue
        if 'distortions' in meta_item:
            for distortion in meta_item['distortions']:
                # 删除指定的字段
                distortion.pop('perception impact', None)
                distortion.pop('position', None)
                distortion.pop('severity', None)
                distortion.pop('visual manifestation', None)
    
    # 将更新后的数据保存到新的文件
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file, 'w') as f:
        json.dump(meta_data, f, indent=4)

# 执行函数
extract_dist_meta(meta_file1, save_file1)
extract_dist_meta(meta_file2, save_file2)
