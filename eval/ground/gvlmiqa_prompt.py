import json
import os
import re
import random
def extract_distortion(data):
    # 提取括号里的内容
    issues = set()
    pattern = r'\((.*?)\)'

    # 遍历 global 和 local 字典
    for key in data.get("global", {}).keys():
        match = re.search(pattern, key)
        if match:
            issues.add(match.group(1))

    for key in data.get("local", {}).keys():
        match = re.search(pattern, key)
        if match:
            issues.add(match.group(1))
    
    return list(issues)

def convert_to_mcq(data):
    new_data = []
    image_folder = "../datasets/images/single_1w"
    # Extract relevant information from the input dictionary
    
    image = os.path.join(image_folder, data["filename"])
    issues = extract_distortion(data)  
    options = ["No", "Yes"]

    for issue in issues:
        random.shuffle(options)
        question = f"Is there any {issue.lower()} issue in the image?\nA. {options[0]}\nB. {options[1]}\nAnswer with the option's letter from the given choices directly."
        if options[0] == 'Yes':
            ans = "A"
        else:
            ans = "B"
        format = {
            "image": image,
            "question": question,
            "correct_ans": ans,
            "type": issue.lower()
        }
        new_data.append(format)

    return new_data

def process_benchmark(json_file='../gen_prompt/dataset/assessment_final_484_主观验证无误.json'):
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    temp_data = []
    processed_data = []
    for item in raw_data:
        new_item = convert_to_mcq(item)
        temp_data.extend(new_item)
    for item in temp_data:
        image_path = item["image"]
        text = item["question"]
        processed_item = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text},
            ],
        }
        processed_data.append(processed_item)
    return temp_data, processed_data

def process_benchmark_ground(json_file='../datasets/val_json/benchmark_2k_normalized.json', image_folder='../datasets/images/gvlmiqa_bench'):
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    vis_ground_text = 'Please identify the quality issues in the image and give their bounding box coordinates both globally and locally.'
    cap_ground_text = 'Describe the image content and the image quality issues with grounding.'
    processed_data = []
    temp_data = []
    for item in raw_data:
        temp_data.append({
            "image": item['filename'],
            "global": item['global'],
            "local": item['local'],
            # "assessment": item['assessment_split']
        })
    for item in temp_data:
        image_path = os.path.join(image_folder, item["image"])
        vis_ground_item = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": vis_ground_text},
            ],
            "type": "vis"
        }
        processed_data.append(vis_ground_item)
        cap_ground_item = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": cap_ground_text},
            ],
            "type": "cap"
        }
        processed_data.append(cap_ground_item)
    return temp_data, processed_data


if __name__ == '__main__':

    raw_data, processed_data = process_benchmark_ground()
    print(raw_data[0:2])
    print(processed_data[0:2])