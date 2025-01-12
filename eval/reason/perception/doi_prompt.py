import json
import os
import re
import random
from prompt import convert_to_mcq

def convert_to_saq(data):
    # Extract relevant information from the input dictionary
    question = data["question"]
    formatted_output = f"You are an expert in image quality assessment. You should give a short answer to the question below:\n{question}\nPlease identify them as precisely as possible, referencing notable features or locations in the image.\nOutput in a list format."
    formatted_output_grounding = f"You are an expert in image quality assessment. You should give a short answer to the question below:\n{question}\nPlease identify them with their bounding box coordinate."
    
    return formatted_output, formatted_output_grounding

def process_benchmark(json_file='data/meta_json/benchmark-v1/test/test_mcq_v1.json', save_path='data/meta_json/benchmark-v1/test/test_mcq_qbench_format_v1.json', image_folder='../datasets/images/doi-images-all'):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 遍历数据并转换为目标格式
    output = []
    for item in data:
        img_path = item["image"]
        mcq = item["mcq"]
        for question_item in mcq:
            candidates = question_item["false candidates"] + [question_item["answer"]]
            random.shuffle(candidates)
            print(candidates)
            # input()
            new_item = {
                'type': question_item["question_type"],  # 根据question_type设定type
                'concern': question_item["concern"],  # concern保留原样
                'question': question_item["question"],  # question保留原样
                'img_path': os.path.join(image_folder, img_path),  # 图片路径
                'candidates': candidates,  # 合并选项
                'correct_ans': question_item["answer"]  # 正确答案
            }
            output.append(new_item)
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)


def process_benchmark_ground(json_file='../gen_prompt/dataset/assessment_final_484_主观验证无误.json', image_folder='../datasets/images/doi-images-all'):
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


def process_benchmark_mcq(json_file='data/meta_json/benchmark-v1/release/mcq.json'):
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    processed_data = []
    for item in raw_data:
        image_path = os.path.join('../datasets/images/doi-images-all', item["img_path"])
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist. Skipping...")
            continue
        text, _ = convert_to_mcq(item)
        processed_item = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text},
            ],
        }
        processed_data.append(processed_item)
    return raw_data, processed_data

def process_benchmark_saq(json_file='data/meta_json/benchmark-v1/release/saq.json'):
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    processed_data = []
    for item in raw_data:
        image_path = os.path.join('../datasets/images/doi-images-all', item["img_path"])
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist. Skipping...")
            continue
        question, question_grounding = convert_to_saq(item)
        processed_item = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
        processed_data.append(processed_item)
    return raw_data, processed_data

if __name__ == '__main__':
    raw_data, processed_data = process_benchmark_mcq()
    print(processed_data[0])
    # process_benchmark()
