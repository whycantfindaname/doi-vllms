import argparse
import base64
import json
import os

import weave
from openai import OpenAI
from PIL import Image
import random
from data_utils import load_json, encode_img, conver_meta_data_to_gpt, clean_text
from collections import Counter


OPENAI_API_KEY = "sk-tE7K8vJ9Dla5zDMx87F9EeB7372340C68067179938991e54"
OPENAI_API_BASE = "https://api.gpt.ge/v1"
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

parser = argparse.ArgumentParser(
    description="To Prompt GPT-4 for Image Quality Assessment"
)
parser.add_argument("--meta_file", type=str, default='data/meta_json/train/release/assess_v1.json')
parser.add_argument("--qa_file", type=str, default='data/meta_json/train/test/test_qa_v1.json')
parser.add_argument("--image_folder", type=str, default='../datasets/images/gvlmiqa_train/')


# Weave will track the inputs, outputs and code of this function
@weave.op()
def gpt4o(query, system_prompt, with_dist):
    if with_dist:
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
                        "description": "The concerns addressed in the question, one of the following: existence, type, position, severity, region quality, region importance, visual manifestation, perception impact."
                        }
                    }
                    },
                    "required": ["question", "answer", "concerns"],
                    "additionalProperties": False
                }
                }
            },
            "required": ["question_answer_pair"],
            "additionalProperties": False
            }
        }
        }
    else:
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
                                    }
                                },
                                "required": ["question", "answer"],
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
    idx_meta_end = 1

    meta_file = args.meta_file
    qa_file = args.qa_file
    image_folder = args.image_folder

    meta_data = load_json(meta_file)
    if os.path.exists(qa_file):
        qa_data = load_json(qa_file)
    else:
        qa_data = []
    print(len(meta_data))
    print(len(qa_data))
    check_flag = check(meta_data, qa_data)
    check_duplicate(meta_data)
    dist_paths_error = []
    
    for idx_meta, meta_item in enumerate(meta_data[idx_meta_start:]):
        img_name = meta_item["image"]
        if img_name in [item["image"] for item in qa_data]:
            # print(f"{img_name} has been generated, skip.")
            continue
        print("=" * 100)
        print(idx_meta + idx_meta_start)
        print(img_name)
        img_path = os.path.join(image_folder, img_name)
        
        try:
            dist_assess = meta_item['assessment']['single distortion assessment']
            assess_query = '[Single Distortion Assessment]\n'
            try:
                for dist_item in dist_assess:
                    dist_name = dist_item['name']
                    dist_assess = clean_text(dist_item['assessment'])
                    assess_query += f'{dist_name}: {dist_assess}\n'
                print(assess_query)
            except Exception as e:
                print(f"Error occurred while processing distortion assessment for {img_name}.")
                print(e)
                continue
            with_dist = True
            
            system_prompt_assess_data = (
                "You are an expert in image quality assessment. Generate multiple question and answer pairs based on the following single distortion assessment provided to you."
                + " The questions should be of the following types: yes-or-no,what, how, why, or where. You may use one or multiple types based on the situation."
                + " Our goal is to evaluate the AI assistant's accuracy in assessing image quality and each type of distortion, so the questions you generate should comprehensively evaluate its capabilities." 
                + " These questions should be distortion-oriented, and for each distortion type, 8 concerns need to be considered:\n"
                + " 1. the existence of the distortion;\n"
                + " 2. the type of the distortion;\n"
                + " 3. the position of the distortion;\n"
                + " 4. the severity of the distortion;\n"
                + " 5. the visual manifestation of the distortion;\n"
                + " 6. the perception impact of the distortion;\n"
                + " 7. the quality of the affected region;\n"
                + " 8. the importance of the affected region.\n"
                + " The questions can align with or distract from the distortion assessment. The number of the questions which align with the assessment should be THE SAME AS the number of the questions which distract from the assessment."
                + " Distracting questions are incorrect questions that directly contradict or misinterpret the distortion information from the 8 concerns by introducing incorrect assumptions or misleading interpretations, thereby challenging the assistant's ability to distinguish between correct and incorrect interpretations of the distortion's impact.\n"
                + " Requirements for the Questions and Answers:\n"
                + " 1. The questions must not reference the information provided directly, but rather focus on the image itself.\n"
                + " 2. Do not deviate from the distortion names or introduce unrelated terms or variations.\n"
                + " 3. You need to generate questions for each distortion type in the information, and each question-answer pair should involve as many concerns as possible. The more concerns addressed in each question, the more informative the question will be.\n"
                + " 4. For each distortion type, generate as few questions as possible while ensuring that all relevant information is covered.\n"
                + " 5. The combinations of concerns addressed in each question need to be arranged flexibly to avoid repetition or overly fixed patterns.\n"
                + " 6. Your answers should not simply restate the information provided. If the question aligns with the distortion assessment, provide reasoning that integrates the information. If the question distracts from the distortion assessment, point out the mistake based on the information.\n"
                + " 7. Aovid generating questions that are essentially the same or similar to each other."
                + " 8. Ensure that the grammar is correct and the question-answer pairs are logically consistent and well-structured.\n"
                + " As you generate the questions, continuously consider whether your question-answer pairs are comprehensive enough to evaluate the AI assistant's ability. "
                + " The number of generated questions should be less than 20."
            )
        except:
            with_dist = False
            overall_assess = meta_item['assessment']
            assess_query = f'[Overall Quality Assessment]\n{overall_assess}'
            system_prompt_assess_data = (
                "You are an expert in image quality assessment. Generate multiple question and answer pairs based on the following overall quality assessment provided to you."
                + " The questions should be of the following types: yes-or-no,what, how, why, or where. You may use one or multiple types based on the situation."
                + " Our goal is to evaluate the AI assistant's accuracy in assessing image quality, so the questions you generate should comprehensively evaluate its capabilities." 
                + " The questions can align with or distract from the distortion assessment. The number of the questions which align with the assessment should be THE SAME AS the number of the questions which distract from the assessment."
                + " Distracting questions are incorrect questions that directly contradict or misinterpret the assessment by introducing incorrect assumptions or misleading interpretations."
                + " When generating questions, pretend that you have only seen the image and have no prior knowledge of the provided information."
                + " Questions should be clear, general, and easy to understand, without being overly specific."
                + " The answers should not simply restate the information provided. If the question aligns with the assessment, provide reasoning that integrates the information. If the question distracts from the assessment, point out the mistake based on the information. "
                + " The questions and answers MUST not reference the assessment provided directly, but rather focus on the image itself."
                + " Ensure that the grammar is correct and that the question-answer pairs are logically consistent and well-structured."
                + " The total number of generated questions should be less than 5."
            )



        try:
            print("Generating question answer pairs...")
            content= gpt4o(assess_query, system_prompt_assess_data, with_dist)
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
