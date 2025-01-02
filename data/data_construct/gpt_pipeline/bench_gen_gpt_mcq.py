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
parser.add_argument("--mcq_file", type=str, default='data/meta_json/benchmark-v1/test/test_mcq_v1.json')
parser.add_argument("--image_folder", type=str, default='../datasets/images/gvlmiqa_bench/')


# Weave will track the inputs, outputs and code of this function
@weave.op()
def gpt4o(img_path, query, system_prompt, example_meta_inputs, example_gpt_output_strs):
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
                    "description": "The correct answer to the question."
                },
                "false candidates": {
                    "type": "array",
                    "items": {
                    "type": "string",
                    "description": "The false candidates for the answer."
                    }
                },
                "concern": {
                    "type": "string",
                    "description": "One of the following concerns addressed in the correct answers and false candidates: existence, type, position, severity, visual manifestation, and perception impact."
                },
                "question_type":{
                    "type": "string",
                    "description": "The type of the question, one of the following: yes-or-no, what, how, why, where."
                },
                "distortion_id":{
                    "type": "array",
                    "items": {
                    "type": "string",
                    "description": "The id of the distortions addressed in the question, if not applicable, return null."
                    }
                }
                },
                "required": ["question", "answer", "concern", "question_type", "false candidates", "distortion_id"],
                "additionalProperties": False
            }
            }
        },
        "required": ["question_answer_pair"],
        "additionalProperties": False
        }
    }
    }
    messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
    for meta_input, gpt_output_str in zip(example_meta_inputs, example_gpt_output_strs):
        messages.append(
            {
                "role": "user",
                "content": meta_input,
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": gpt_output_str,
            }
        )
    messages.append(
        {
            "role": "user",
            "content": query,
        }
    )
    resp = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=messages,
        temperature=0.5,
        response_format=resp_format
    )
    print(resp.usage)
    content = resp.choices[0].message.content
    return content

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
    idx_meta_start = 2
    idx_meta_end = 3

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
    example_meta_inputs = [convert_meta_item_to_gpt(mcq_data[0]), convert_meta_item_to_gpt(mcq_data[1])]
    example_gpt_outputs = [{"question_answer_pair": mcq_data[0]['mcq']}, {"question_answer_pair": mcq_data[1]['mcq']}]
    example_gpt_output_strs = [json.dumps(example_gpt_output) for example_gpt_output in example_gpt_outputs]
    
    
    for idx_meta, meta_item in enumerate(meta_data[idx_meta_start:]):
        img_name = meta_item["image"]
        if img_name in [item["image"] for item in mcq_data]:
            print(f"{img_name} has been generated, skip.")
            continue
        print("=" * 100)
        print(idx_meta + idx_meta_start)
        print(img_name)
        img_path = os.path.join(image_folder, img_name)
        
        assess_query = convert_meta_item_to_gpt(meta_item)
        
        system_prompt_assess_data = (
            "You are an expert in image quality assessment. Your task is to generate comprehensive question-and-answer pairs based on the provided image and its associated distortion information. The distortion information includes:\n"
            + " - **Distortion ID**\n"
            + " - **Distortion Type**\n"
            + " - **Position**\n"
            + " - **Severity**\n"
            + " - **Visual Manifestation**\n"
            + " - **Perception Impact**\n"
            + " #### Question Types\n"
            + " Generate questions of the following types: **yes-or-no**, **what**, **how**, **why**, or **where**. Use one or multiple types as appropriate for each distortion.\n"
            + " #### Goal\n"
            + " The primary goal is to evaluate the AI assistant's ability to accurately identify and assess distortions. Your questions and answers should comprehensively test its understanding and reasoning capabilities.\n"
            + " #### Key Areas of Concern\n"
            + " For each distortion, consider the following six aspects in the generated answers and generate few question-answer pairs per distortion:\n"
            + " 1. The existence of the distortion.\n"
            + " 2. The type of the distortion.\n"
            + " 3. The position of the distortion.\n"
            + " 4. The severity of the distortion.\n"
            + " 5. The visual manifestation of the distortion.\n"
            + " 6. The perception impact of the distortion.\n"
            + " #### Requirements for Questions and Answers\n"
            + " 1. **Clarity**: Assume no prior knowledge of the distortion information and ensure questions are clear, easy to understand and specific enough to relate to the distortion being evaluated.\n"
            + " 2. **Relevance**: Do not introduce unrelated terms or deviate from the provided distortion types.\n"
            + " 3. **Exhaustiveness**: Address every distortion in the information, with few question-answer pairs per distortion.\n"
            + " 4. **Conciseness and Core Information**: Answers shoule be concise and contain only the core information, with minimum words."
            + " 5. **Focus**: Correct answer of each question-answer pair should target only one of the six aspects.\n"
            + " 6. **False Candidates**: Provide several false answers under the key 'false candidates' for each question. These must:\n"
            + "    - Appear reasonable based on the question.\n"
            + "    - Have the same format and subject as the correct answer.\n"
            + "    - Contradict the provided image and distortion information.\n"
            + "    - Include misleading elements by mixing characteristics from other distortions, challenging the assistant to correctly identify the specific distortion being evaluated.\n"
            + " 7. **Logical Consistency**: Ensure all questions and answers are grammatically correct, logically consistent, and mutually non-contradictory.\n"
            + " 8. **Question Number**: Limit the total number of question-answer pairs to fewer than 15. Prioritize key distortions if necessary.\n"
            + " As you generate the question-answer pairs, ensure they are comprehensive enough to evaluate the AI assistant's understanding and reasoning. Your output should include logically structured question-answer pairs with all requirements met."
        )


        try:
            print("Generating question answer pairs...")
            content= gpt4o(img_path, assess_query, system_prompt_assess_data, example_meta_inputs, example_gpt_output_strs)
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
