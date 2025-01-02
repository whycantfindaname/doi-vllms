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

parser = argparse.ArgumentParser(
    description="To Prompt GPT-4 for Image Quality Assessment"
)
parser.add_argument("--meta_file", type=str, default='data/meta_json/benchmark-v1/release/dist_info_v1.json')
parser.add_argument("--qa_file", type=str, default='data/meta_json/benchmark-v1/test/test_qa_assess_v1.json')
parser.add_argument("--image_folder", type=str, default='../datasets/images/gvlmiqa_bench/')

if __name__ == "__main__":
    weave.init("image quality assessment")
    args = parser.parse_args()
    idx_meta_start = 1
    idx_meta_end = 2

    meta_file = args.meta_file
    qa_file = args.qa_file
    image_folder = args.image_folder

    meta_data = load_json(meta_file)
    if os.path.exists(qa_file):
        qa_data = load_json(qa_file)
    else:
        qa_data = []

    check_flag = check(meta_data, qa_data)
    check_duplicate(meta_data)
    dist_paths_error = []
    example_meta_input = convert_meta_item_to_gpt(qa_data[0])
    example_gpt_output = {"question_answer_pair": qa_data[0]['qa-pairs']}
    example_gpt_output_str = json.dumps(example_gpt_output)
    print(example_meta_input)
    print(example_gpt_output_str)
    
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
                        "description": "The corresponding answer to the question."
                    },
                    "concerns": {
                        "type": "array",
                        "items": {
                        "type": "string",
                        "description": "Multiple of the following concerns addressed in the question: existence, type, position, severity, visual manifestation, and perception impact."
                        },
                    },
                    "question_types":{
                        "type": "array",
                        "items": {
                        "type": "string",
                        "description": "One or mulitple of the following: yes-or-no, what, how, why, where."
                        }
                    },
                    "distortion_id":{
                        "type": "string",
                        "description": "The id of the distortion addressed in the question."
                    }
                    },
                    "required": ["question", "answer", "concerns", "question_types", "distortion_id"],
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
                        {"type": "text", "text": example_meta_input},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": example_gpt_output_str},
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{img_base64}"
                                },
                        },
                    ],
                }
            ],
            temperature=0.5,
            response_format=resp_format
        )
        print(resp.usage)
        content = resp.choices[0].message.content
        return content
    
    for idx_meta, meta_item in enumerate(meta_data[idx_meta_start:]):
        img_name = meta_item["image"]
        if img_name in [item["image"] for item in qa_data]:
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
            + " For each distortion, consider the following eight aspects and generate one question-answer pair that evaluates multiple aspects:\n"
            + " 1. The existence of the distortion.\n"
            + " 2. The type of the distortion.\n"
            + " 3. The position of the distortion.\n"
            + " 4. The severity of the distortion.\n"
            + " 5. The visual manifestation of the distortion.\n"
            + " 6. The perception impact of the distortion.\n"
            + " #### Requirements for Questions and Answers\n"
            + " 1. **Clarity and Generality**: Pretend you have only seen the image and have no prior knowledge of the provided distortion information. Ensure questions are clear, general, and easy to understand. Avoid overly specific or technical phrasing.\n"
            + " 2. **Concise and Detailed Answers**: Answers should be concise yet detailed, fully leveraging the provided distortion information. Do not introduce unrelated terms or deviate from the given distortion types.\n"
            + " 3. **Exhaustiveness**: Do not omit any distortion addressed in the provided information. Each distortion should have ONLY ONE question-answer pair.\n"
            + " 4. **Distractors**: Include a few misleading or conflicting questions to test the assistant's reasoning. These questions should subtly misinterpret the distortion information, requiring the assistant to reason and provide a correction in the answer.\n"
            + " 5. **Logical Consistency**: Ensure all question-answer pairs are grammatically correct, logically consistent, and do not contradict each other.\n"
            + " 6. **Reasoned Answers**: If a question aligns with the provided information, integrate the details into the answer. If a question is misleading or incorrect, explicitly highlight the issue and provide reasoning.\n"
            + " 7. **Flexible Arrangement**: The combinations of concerns addressed in each question answer pair need to be arranged flexibly to avoid overly fixed patterns.\n"
            + " #### Comprehensive Evaluation\n"
            + " As you generate the question-answer pairs, continuously evaluate whether they are comprehensive enough to test the AI assistant's ability to:\n"
            + " 1. Interpret distortion information.\n"
            + " 2. Reason through misleading or conflicting questions.\n"
            + " 3. Provide accurate and detailed assessments of distortions.\n"
            + " Your output should consist of logically structured and well-written question-answer pairs, ensuring all requirements are met.\n"
            + " The number of question-answer pairs should be the same as the number of distortions in the provided information."
        )


        try:
            print("Generating question answer pairs...")
            content= gpt4o(img_path, assess_query, system_prompt_assess_data)
            print(content)
            resp_dict = json.loads(content)
            meta_item["qa-pairs"] = resp_dict['question_answer_pair']
            qa_data.append(meta_item)
            print(f"QA-pairs for {img_name} generated.")
            with open(qa_file, "w") as fw:
                json.dump(qa_data, fw, indent=4, ensure_ascii=False)
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
    fail_dir = os.path.dirname(qa_file)
    os.makedirs(fail_dir, exist_ok=True)
    fail_path = os.path.join(fail_dir, "res_fail.txt")
    with open(fail_path, "w") as fw:
        fw.write("\n".join(dist_paths_error))
