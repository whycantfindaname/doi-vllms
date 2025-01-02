import argparse
import base64
import json
import os

import weave
from openai import OpenAI
from PIL import Image
import random
from data_utils import load_json, encode_img, conver_meta_data_to_gpt
from collections import Counter

OPENAI_API_KEY = "sk-tE7K8vJ9Dla5zDMx87F9EeB7372340C68067179938991e54"
OPENAI_API_BASE = "https://api.gpt.ge/v1"
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

parser = argparse.ArgumentParser(
    description="To Prompt GPT-4 for Image Quality Assessment"
)
parser.add_argument("--meta_file", type=str, default='data/meta_json/benchmark-v1/release/dist_info_v1.json')
parser.add_argument("--vqa_file", type=str, default='data/meta_json/benchmark-v1/test/test_vqa_v1.json')
parser.add_argument("--image_folder", type=str, default='../datasets/images/gvlmiqa_bench/')

def convert_meta_item_to_gpt(meta_item):
    if isinstance(meta_item['distortions'], list):
        assess_query = '[Distortion Infomation]\n'
        for distortion in meta_item['distortions']:
            id = distortion['id']
            dist = distortion['distortion']
            position = distortion['position']
            severity = distortion['severity']
            visual_manifestation = distortion['visual manifestation']
            perception_impact = distortion['perception impact']
            assess_query += f'<{id}> Distortion: {dist}, Position: {position}, Severity: {severity}, Visual Manifestation: {visual_manifestation}, Perception Impact: {perception_impact}\n'
    return assess_query
# Weave will track the inputs, outputs and code of this function
@weave.op()
def gpt4o(img_path, query, system_prompt):
    mime_type, img_base64 = encode_img(img_path)
    resp_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "question_answer_pairs",
        "strict": True,
        "schema": {
        "type": "object",
        "properties": {
            "question_answer_pair": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                "question": {
                    "type": "string",
                    "description": "The generated question."
                },
                "answer": {
                    "type": "string",
                    "description": "The corresponding answer to the question: 'Yes' or 'No'."
                },
                "concerns": {
                    "type": "string",
                    "description": "One of the following concerns addressed in the question: existence, type, position, severity, visual manifestation, and perception impact."
                },
                "distortion_id":{
                        "type": "string",
                        "description": "The id of the significant distortion addressed in the question."
                    }
                },
                "required": ["question", "answer", "concerns", "distortion_id"],
                "additionalProperties": False
            }
            }
        },
        "required": ["question_answer_pair"],
        "additionalProperties": False
        }
    }
    }

    resp = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{img_base64}"
                        }
                    }
                ],
            }
        ],
        temperature=0.5,
        response_format=resp_format
    )
    print(resp.usage)
    content = resp.choices[0].message.content
    return content

def check(meta_data, assess_data):
    # check if all images have been generated
    img_names = [item["image"] for item in meta_data]
    assess_names = [item["image"] for item in assess_data]
    if set(img_names) == set(assess_names):
        print("All images have been generated.")
    else:    
        len_diff = len(img_names) - len(assess_names)
        if len_diff > 0:
            print(f"There are {len_diff} images not generated.")
def check_duplicate(meta_data):
    # check if there are duplicate images
    file_names = [item["image"] for item in meta_data]
    file_name_counts = Counter(file_names)
    duplicates = {name: count for name, count in file_name_counts.items() if count > 1}
    if duplicates:
        print("Duplicate images found:")
        for name, count in duplicates.items():
            print(f"{name}: {count} times")
    else:
        print("No duplicate images found.")

if __name__ == "__main__":
    weave.init("image quality assessment")
    args = parser.parse_args()
    idx_meta_start = 0
    idx_meta_end = 1

    meta_file = args.meta_file
    vqa_file = args.vqa_file
    image_folder = args.image_folder

    meta_data = load_json(meta_file)
    if os.path.exists(vqa_file):
        vqa_data = load_json(vqa_file)
    else:
        vqa_data = []

    check_flag = check(meta_data, vqa_data)
    check_duplicate(meta_data)
    dist_paths_error = []
    
    for idx_meta, meta_item in enumerate(meta_data[idx_meta_start:]):
        img_name = meta_item["image"]
        if img_name in [item["image"] for item in vqa_data]:
            print(f"{img_name} has been generated, skip.")
            continue
        print("=" * 100)
        print(idx_meta + idx_meta_start)
        print(img_name)
        img_path = os.path.join(image_folder, img_name)
        
        assess_query = convert_meta_item_to_gpt(meta_item)
        
        system_prompt_assess_data = (
            "You are an expert in image quality assessment. Generate multiple yes-or-no question-answer pairs based on the provided image and its associated distortion information."
            + " The primary goal is to evaluate the AI assistant's ability to accurately identify and assess distortions." 
            + " Your questions and answers should comprehensively test its understanding and reasoning capabilities. These questions should be distortion-oriented, and for each distortion, 6 concerns need to be considered:\n"
            + " 1. the existence of the distortion;\n2. the type of the distortion;\n3. the position of the distortion;\n4. the severity of the distortion;\n5. the visual manifestation of the distortion;\n6. the perception impact of the distortion.\n"
            + " The questions can align with or distract from the distortion information. The number of the questions which align with the information should be THE SAME AS the number of the questions which distract from the information.\n" 
            + " Distracting questions are incorrect questions that directly contradict or misinterpret the distortion information from one of the 6 concerns by introducing incorrect assumptions or misleading interpretations, thereby challenging the assistant's ability to distinguish between correct and incorrect interpretations of the distortion.\n" 
            + " Answering Aligning Questions: The answer should be 'Yes.'\nAnswering Distracting Questions: The answer should be 'No.'\n" 
            + " Requirements for the Questions and Answers:\n1. Do not omit any distortion addressed in the provided information.\n2. Each distortion should have ONLY ONE question-answer pair.\n3. Each question-answer pair should address ONLY ONE concern of the 6 concerns.\n4. Do not introduce unrelated terms or deviate from the given distortion types.\n5. Pretend you have only seen the image and have no prior knowledge of the provided distortion information. Ensure questions are clear, general, and easy to understand. Avoid overly specific or technical phrasing.\n6. Avoid generating questions that are essentially the same or too similar.\n7. The number of question-answer pairs should be less than 5." 
            + " As you generate the questions, continuously consider whether your question-answer pairs are comprehensive enough to evaluate the AI assistant's ability."
        )


        try:
            print("Generating vqa pairs...")
            content= gpt4o(img_path, assess_query, system_prompt_assess_data)
            print(content)
            resp_dict = json.loads(content)
            meta_item["vqa"] = resp_dict['question_answer_pair']
            vqa_data.append(meta_item)
            print(f"QA-pairs for {img_name} generated.")
            with open(vqa_file, "w") as fw:
                json.dump(vqa_data, fw, indent=4, ensure_ascii=False)
        except:
            import sys
            print("Error occurred while generating assessment for {}.".format(img_name))
            except_type, except_value, except_traceback = sys.exc_info()
            except_file = os.path.split(except_traceback.tb_frame.f_code.co_filename)[1]
            exc_dict = {
                "error type": except_type,
                "error info": except_value,
                "error file": except_file,
                "error line": except_traceback.tb_lineno,
            }
            print(exc_dict)
            dist_paths_error.append(img_name)
    fail_dir = os.path.dirname(vqa_file)
    os.makedirs(fail_dir, exist_ok=True)
    fail_path = os.path.join(fail_dir, "res_fail.txt")
    with open(fail_path, "w") as fw:
        fw.write("\n".join(dist_paths_error))
