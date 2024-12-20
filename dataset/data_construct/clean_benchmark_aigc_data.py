import json
import os 
from collections import defaultdict
def clean_aigc_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(len(data))
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            if "AIGC" in item['image']:
                data.remove(item)
        print(len(data))
        json.dump(data, f, ensure_ascii=False, indent=4)
import json
from collections import defaultdict

def clean_dist(file, output_file):
    # 读取文件
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 遍历每个 item
    for item in data:
        # 过滤掉没有 "distortion" 的 bboxes
        filtered_bboxes = []
        for bbox in item.get("bboxes", []):
            if "distortion" not in bbox:
                # 跳过无效的 bbox
                continue
            distortion = bbox.get("distortion", "unknown")
            
            # 修正 distortion 值
            if distortion == "Excessive darkness":
                bbox["distortion"] = "Underexposure"
                distortion = "Underexposure"
            
            # 跳过指定的 distortion
            if distortion in ["Interlaced scanning", "Banding effect", "Moiré pattern"]:
                continue

            if distortion == "Low clarity":
                if bbox["severity"] in ['trival', 'minor']:
                    continue
            
            if distortion == "Edge aliasing effect":
                if bbox["severity"] in ['trival']:
                    continue
            
            # 只保留有效 bbox
            filtered_bboxes.append(bbox)
        
        # 更新 item["bboxes"]，只保留有效的 bbox
        item["bboxes"] = filtered_bboxes
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def show_number(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 动态记录 distortion 出现个数
    distortion_count = defaultdict(int)
    for item in data:
        for bbox in item.get("bboxes", []):
            distortion = bbox.get("distortion", "unknown")
            distortion_count[distortion] += 1

    # 输出结果
    print("Distortion count:")
    for distortion, count in distortion_count.items():
        print(f"{distortion}: {count}")

def check_sanity(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        mos = item['mos']
        bboxes = item.get("bboxes", [])
        if mos < 4.2 and len(bboxes) == 0:
            print(item)
            data.remove(item)
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
            
if __name__ == "__main__":
    file1 = "dataset/meta_json/benchmark/benchmark_2k_normalized_new.json"
    file2 = "dataset/meta_json/benchmark/benchmark_2k_normalized_v1.json"
    file3 = "dataset/meta_json/benchmark/benchmark_2k_normalized_v2.json"
    # clean_aigc_data(input_file, output_file)
    # print("before clean:")
    # show_dist_number(input_file)
    # print("after clean:")
    # clean_dist(file2, file3)
    show_number(file3)
    check_sanity(file3)