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
parser.add_argument("--qa_file", type=str, default='data/meta_json/benchmark/test/test_dist_qa_assess_v2.json')
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
                "concerns": {
                    "type": "array",
                    "items": {
                    "type": "string",
                    "description": "The concerns addressed in the question-answer pair, one of the following: existence, type, position, severity, region quality, region importance, visual manifestation, perception impact."
                    }
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
                    "description": "The id of the significant distortion addressed in the question."
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
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    # {
                    #     "type": "image_url",
                    #     "image_url": {
                    #         "url": f"data:{mime_type};base64,{img_base64}"
                    #         },
                    # },
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
    idx_meta_end = -1

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
    
    for idx_meta, meta_item in enumerate(meta_data[idx_meta_start:idx_meta_end]):
        img_name = meta_item["image"]
        if img_name in [item["image"] for item in qa_data]:
            print(f"{img_name} has been generated, skip.")
            continue
        print("=" * 100)
        print(idx_meta + idx_meta_start)
        print(img_name)
        img_path = os.path.join(image_folder, img_name)
        
        if isinstance(meta_item['assessment'], dict):
            significance_assess = meta_item['assessment']['overall quality assessment'].split('\n')[-2] + meta_item['assessment']['overall quality assessment'].split('\n')[-1]
            assess_query = f'[Distortion Significance Assessment]\n{significance_assess}\n'
            assess_query += '[Distortion Infomation]\n'
            for distortion in meta_item['distortions']:
                id = distortion['id']
                dist = distortion['distortion']
                position = distortion['position']
                severity = distortion['severity']
                visual_manifestation = distortion['visual manifestation']
                perception_impact = distortion['perception impact']
                region_quality = distortion['region_quality_scores']
                region_importance = distortion['region_importance_score']
                assess_query += f'<{id}> Distortion: {dist}, Position: {position}, Severity: {severity}, Visual Manifestation: {visual_manifestation}, Perception Impact: {perception_impact}, Region Quality: {region_quality}, Region Importance: {region_importance}\n'
        print(assess_query)

        
        system_prompt_assess_data = (
            "You are an expert in image quality assessment. Generate multiple question and answer pairs based on the following distortion significance assessment and distortion information provided to you."
            + " The distortion information includes the distortion id, distortion name, position, severity, region quality, region importance, visual manifestation and perception impact."
            + " The questions should be of the following types: yes-or-no,what, how, why, or where. You may use one or multiple types based on the situation."
            + " Our goal is to evaluate the AI assistant's accuracy in identifying and assessing significant distortions, so the question-answer pairs you generate should comprehensively evaluate its capabilities." 
            + " To accomplish this, you should first identify all significant distortions from the distortion significance assessment and extract the corresponding information from the distortion information. "
            + " Use these information to generate distortion-oriented questions, and for each significant distortion, 8 concerns need to be considered:\n"
            + " 1. the existence of the distortion;\n"
            + " 2. the type of the distortion;\n"
            + " 3. the position of the distortion;\n"
            + " 4. the severity of the distortion;\n"
            + " 5. the visual manifestation of the distortion;\n"
            + " 6. the perception impact of the distortion;\n"
            + " 7. the quality of the affected region;\n"
            + " 8. the importance of the affected region.\n"
            + " Requirements for the Questions and Answers:\n"
            + " 1. When generating questions, pretend that you have only seen the image and have no prior knowledge of the provided information. So the questions should be clear, general, and easy to understand, without being overly specific.\n"
            + " 2. The answers should be concise and as detailed as possible, fully exploiting the providing information. Do not deviate from the distortion names or introduce unrelated terms or variations.\n"
            + " 3. For each significant distortion, strive for brevity: Generate as few question-answer pairs as possible while covering all concerns. Each question-answer pair should address multiple concerns of the distortion.\n"
            + " 4. Ensure flexibility in the combination of concerns addressed in each question-answer pair to avoid repetition and rigid patterns.\n"
            + " 5. Generate some distracting questions. These questions are designed to challenge the AI assistant's ability to accurately identify and assess significant distortions. They should introduce misleading or conflicting information that tests the assistant's reasoning and ability to correctly interpret distortion details.\n"
            + " 5. Provide reasoned answers. If the question aligns with the provided information, integrate the information into your answer. If the question misinterprets the information, point out the mistake and reason according to the assessment.\n"
            + " 6. Avoid generating question-answer pairs that contradict each other.\n"
            + " 7. Ensure that the grammar is correct and the question-answer pairs are logically consistent and well-structured.\n"
            + " As you generate the questions, continuously consider whether your question-answer pairs are comprehensive enough to evaluate the AI assistant's ability. "
            + " The number of generated question-answer pairs should be less than 8."
        )



        try:
            print("Generating question answer pairs...")
            content= gpt4o(assess_query, system_prompt_assess_data)
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
