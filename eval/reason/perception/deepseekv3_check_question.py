import argparse
import json
import os
from openai import OpenAI
import base64
from PIL import Image
from tqdm import tqdm
from doi_prompt import process_benchmark_mcq
OPENAI_API_KEY = "sk-cd17cf770bbd407bb02a375f562ecf77"
OPENAI_API_BASE = "https://api.deepseek.com"
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)


def deepseek(query):

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an expert in image quality assessment. You need to answer the following multiple-choice question without the image. This question only has one correct answer. Please select the correct answer."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                ],
            }
        ],
        stream=False
    )
    print(resp.usage)
    content = resp.choices[0].message.content
    return content

if __name__ == "__main__":
    save_dir = 'data/meta_json/benchmark-v1/process'
    save_name = 'deepseekv3_check_text-only.json'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    # raw_data, processed_data = process_qbench()
    raw_data, processed_data = process_benchmark_mcq('data/meta_json/benchmark-v1/test/test_perception_position_qbench_format.json')
    count = 0
    dist_paths_error = []
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            save_data = json.load(f)
    else:
        save_data = []
    # 保留包含 'response' 键的项
    save_data = [item for item in save_data if 'response' in item.keys()]

    print(len(save_data))

    for gt, data in tqdm(zip(save_data,processed_data), total=len(save_data)):
        img_path = data['content'][0]['image']
        query = data['content'][1]['text']
        # if save_data and os.path.basename(img_path) in [d['img_path'] for d in save_data]:
        #     # print(f"Skip {img_path}")
        #     continue
        print(query)
        try:
            answer = deepseek(query)
            print(answer)
            if isinstance(gt['response'], str):
                gt['response'] = [gt['response']]
            gt['response'].append(answer)
            print(gt['response'])
            save_data.append(gt)
            # count += 1
            # if count % 100 == 0:
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
        except:
            import sys

            except_type, except_value, except_traceback = sys.exc_info()
            except_file = os.path.split(except_traceback.tb_frame.f_code.co_filename)[1]
            exc_dict = {
                "error type": except_type,
                "error info": except_value,
                "error file": except_file,
                "error line": except_traceback.tb_lineno,
            }
            print(exc_dict)
            dist_paths_error.append(img_path)
