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
parser.add_argument("--meta_file", type=str, default='data/meta_json/benchmark/test/test_assess_v1.json')
parser.add_argument("--mcq_file", type=str, default='data/meta_json/benchmark/test/test_dist_mcq_assess_v1.json')
parser.add_argument("--image_folder", type=str, default='../datasets/images/gvlmiqa_bench/')


# Weave will track the inputs, outputs and code of this function
@weave.op()
def gpt4o(query, system_prompt):
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
                    "description": "The corresponding answer to the question."
                },
                "false candidates": {
                    "type": "array",
                    "items": {
                    "type": "string",
                    "description": "The false candidates for the answer."
                    }
                },
                "concerns": {
                    "type": "array",
                    "items": {
                    "type": "string",
                    "description": "The concerns addressed in the question, one of the following: existence, position, severity, region quality, region importance, visual manifestation, perception impact."
                    }
                },
                "question_type":{
                    "type": "string",
                    "description": "The type of the question, one of the following: what, how, why, where."
                }
                },
                "required": ["question", "answer", "concerns", "question_type", "false candidates"],
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
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query}
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
    mcq_file = args.mcq_file
    image_folder = args.image_folder

    meta_data = load_json(meta_file)
    if os.path.exists(mcq_file):
        mcq_data = load_json(mcq_file)
    else:
        mcq_data = []

    check_flag = check(meta_data, mcq_data)
    check_duplicate(meta_data)
    dist_paths_error = []
    
    for idx_meta, meta_item in enumerate(meta_data[idx_meta_start:idx_meta_end]):
        img_name = meta_item["image"]
        if img_name in [item["image"] for item in mcq_data]:
            print(f"{img_name} has been generated, skip.")
            continue
        print("=" * 100)
        print(idx_meta + idx_meta_start)
        print(img_name)
        img_path = os.path.join(image_folder, img_name)
        
        overall_assess = meta_item['assessment']['overall quality assessment']
        assess_query = f'[Overall Quality Assessment]\n{overall_assess}'
        
        system_prompt_assess_data = (
            "You are an expert in image quality assessment. Generate multiple what/how/why/where question and answer pairs based on the following overall image quality assessment provided to you.\n"
            + " Our goal is to evaluate the AI assistant’s accuracy in assessing image quality and various types of distortions, so the questions you generate should comprehensively test its capabilities.\n"
            + " These questions should be distortion-oriented, and for each distortion type, 7 concerns need to be considered:\n"
            + " 1. the existence of the distortion;\n"
            + " 2. the position of the distortion;\n"
            + " 3. the severity of the distortion;\n"
            + " 4. the visual manifestation of the distortion;\n"
            + " 5. the perception impact of the distortion;\n"
            + " 6. the quality of the affected region;\n"
            + " 7. the importance of the affected region.\n"
            + " Requirements for the Questions and Answers:\n"
            + " 1. Do not miss any distortion type in the assessment.\n"
            + " 2. Do not introduce unrelated terms or variations in the type of distortion.\n"
            + " 3. Avoid generating questions that are essentially the same or too similar.\n"
            + " 4. The questions should be clear, concise, and easy to understand.\n"
            + " 5. Ensure that the answers are concise and contain only the core information, with minimum words.\n"
            + " 6. Each question should have several false answers under the key of 'false candidates', which should appear reasonable given the question but contradict the assessment.\n"
            + " As you generate the questions, continually evaluate whether your question-answer pairs are comprehensive enough to thoroughly assess the assistant’s ability."
        )

        try:
            print("Generating question answer pairs...")
            content= gpt4o(assess_query, system_prompt_assess_data)
            print(content)
            resp_dict = json.loads(content)
            meta_item["mcq"] = resp_dict["question_answer_pair"]
            mcq_data.append(meta_item)
            print(f"QA-pairs for {img_name} generated.")
            with open(mcq_file, "w") as fw:
                json.dump(mcq_data, fw, indent=4, ensure_ascii=False)
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
    fail_dir = os.path.dirname(mcq_file)
    os.makedirs(fail_dir, exist_ok=True)
    fail_path = os.path.join(fail_dir, "res_fail.txt")
    with open(fail_path, "w") as fw:
        fw.write("\n".join(dist_paths_error))
