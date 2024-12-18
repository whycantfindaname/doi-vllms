import json
import os
import re
import random
issue_categories = [
    "Meaningless solid color",  # 纯色无意义
    "Edge aliasing effect",  # 锯齿
    "Low clarity",  # 清晰度低
    "Excessive darkness",  # 过暗
    "Blocking artifacts",  # 压缩失真块效应,后续改为compression artifacts
    "Out of focus blur",  # 对焦模糊
    "Overexposure",  # 过曝
    "Noise",  # 噪点
    "Motion blur",  # 运动模糊
    "Underexposure",  # 欠曝
    "Interlaced scanning",  # 隔行扫描
    "Edge ringing Effect",  # 振铃效应
    "Moiré pattern",  # 摩尔纹
    "Banding effect",  # 条带
]

def extract_gt_false_issues(data):
    # Extract issues from global and local keys
    issues = set()
    pattern = r'\((.*?)\)'  # Pattern to extract content inside parentheses

    # Check global and local issues
    for key in data.get("global", {}).keys():
        match = re.search(pattern, key)
        if match:
            issues.add(match.group(1))

    for key in data.get("local", {}).keys():
        match = re.search(pattern, key)
        if match:
            issues.add(match.group(1))
    
    gt_issues = list(issues)  # Ground truth issues list
    
    # Select a number of false issues based on the length of gt_issues
    num_false_issues = len(gt_issues)
    false_issues = random.sample([issue for issue in issue_categories if issue not in gt_issues], num_false_issues)
    
    return gt_issues, false_issues


def convert_to_mcq(data):
    new_data = []
    image_folder = "../datasets/images/single_1w"
    # Extract relevant information from the input dictionary
    
    image = os.path.join(image_folder, data["filename"])
    gt_issues, false_issues = extract_gt_false_issues(data)  
    options = ["No", "Yes"]

    for issue in gt_issues:
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

    for issue in false_issues:
        random.shuffle(options)
        question = f"Is there any {issue.lower()} issue in the image?\nA. {options[0]}\nB. {options[1]}\nAnswer with the option's letter from the given choices directly."
        if options[0] == 'No':
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

def process_benchmark_ground(json_file='../gen_prompt/dataset/assessment_final_484_主观验证无误.json', image_folder='../datasets/images/single_1w'):
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
            "assessment": item['assessment_split']
        })
    for item in temp_data:
        image_path = os.path.join(image_folder, item["image"])
        vis_ground_item = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": vis_ground_text},
            ],
        }
        processed_data.append(vis_ground_item)
        cap_ground_item = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": cap_ground_text},
            ],
        }
        processed_data.append(cap_ground_item)
    return temp_data, processed_data


if __name__ == '__main__':

    raw_data, processed_data = process_benchmark()
    print(len(processed_data))