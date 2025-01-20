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
from typing import Optional


OPENAI_API_KEY = "sk-tE7K8vJ9Dla5zDMx87F9EeB7372340C68067179938991e54"
OPENAI_API_BASE = "https://api.gpt.ge/v1"
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

parser = argparse.ArgumentParser(
    description="To Prompt GPT-4 for Image Quality Assessment"
)
parser.add_argument("--meta_file", type=str, default='data/meta_json/benchmark-v1/test/test_dist_info_v2.json')
parser.add_argument("--position_file", type=str, default='data/meta_json/benchmark-v1/test/test_perception_position.json')
parser.add_argument("--image_folder", type=str, default='../datasets/images/doi-images-all/')
parser.add_argument("--desp_file", type=str, default='data/meta_json/description.json')


# Weave will track the inputs, outputs and code of this function
# @weave.op()
def gpt4o(query, system_prompt, example_meta_inputs: Optional[str]=None, example_gpt_output_strs: Optional[str]=None):
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
                }
                },
                "required": ["question", "answer", "false candidates"],
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
        temperature=0.8,
        response_format=resp_format
    )
    print(resp.usage)
    content = resp.choices[0].message.content
    return content


def convert_meta_item_to_gpt(meta_item):
    if isinstance(meta_item['distortions'], list):
        assess_query = '[Distortion Information]\n'
        # Dictionary to group positions by distortion type
        distortion_positions = {}

        for distortion in meta_item['distortions']:
            dist = distortion['distortion']
            position = distortion['position']
            # Group positions by distortion type
            if dist not in distortion_positions:
                distortion_positions[dist] = []
            distortion_positions[dist].append(position)

        # Construct query with grouped positions
        for distortion, positions in distortion_positions.items():
            positions_str = ' '.join(
                f"{i+1}.{position}" for i, position in enumerate(positions)
            )
            assess_query += f'Distortion: {distortion}, Positions: {positions_str}\n'
            
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
    # weave.init("image quality assessment")
    args = parser.parse_args()
    idx_meta_start = 0
    idx_meta_end = -1

    meta_file = args.meta_file
    position_file = args.position_file
    image_folder = args.image_folder
    desp_file = args.desp_file
    
    if os.path.exists(desp_file):
        desp_data = load_json(desp_file)
    else:
        print("Please generate description first")

    meta_data = load_json(meta_file)
    if os.path.exists(position_file):
        position_data = load_json(position_file)
    else:
        position_data = []

    check_flag = check(meta_data, position_data)
    check_duplicate(meta_data)
    dist_paths_error = []
    
    example_description = "The image shows a person's gloved hand next to a block of butter or cheese on a wooden cutting board. There is Chinese text in the image, which translates to 'Today we are making exploded cheese sandwich.'"
    example_meta_inputs = [f"[Image Information]\n{example_description}\n" + convert_meta_item_to_gpt(position_data[0])]
    example_gpt_outputs = [{"question_answer_pair": position_data[0]['position']}]
    example_gpt_output_strs = [json.dumps(example_gpt_output) for example_gpt_output in example_gpt_outputs]
    print(example_meta_inputs)
    print(example_gpt_output_strs)
    
    for idx_meta, meta_item in enumerate(meta_data[idx_meta_start:]):
        img_name = meta_item["image"]
        if img_name in [item["image"] for item in position_data]:
            print(f"{img_name} has been generated, skip.")
            continue
        print("=" * 100)
        print(idx_meta + idx_meta_start)
        print(img_name)
        img_path = os.path.join(image_folder, img_name)
        
        desp_item = next(
            (item for item in desp_data  if item["filename"] == img_name), None
        )
        if desp_item:
            description_parts = desp_item["gpt4v_description"].split("\n\n")
            description_parts = (
                description_parts
                if len(description_parts) > 1
                else desp_item["gpt4v_description"].split("\n")
            )
            description = (
                description_parts[1]
                .replace("**Answer:** ", "")
                .replace("**Answer:**", "")
                .replace("Answer: ", "")
                .strip()  # This removes leading or trailing spaces
                if len(description_parts) > 1
                else description_parts
            )
        
        if isinstance(description, list):
            query = f"[Image Information]\n{description[0]}\n" + convert_meta_item_to_gpt(meta_item)
        else:
            query = f"[Image Information]\n{description}\n" + convert_meta_item_to_gpt(meta_item)
        print(query)
        system_prompt = (
            "You are an expert in image quality assessment. Your task is to generate comprehensive question, answer and false candidates pairs based on the image information and its associated distortion information."
            + " Image information is a description of the image. Distortion information is a list of distortions and their positions in the image."
            + " #### Goal\n"
            + " The primary goal is to evaluate the AI assistant's ability to accurately identify distortions and their locations. The question, answer and false candidates pairs will be further converted to a multiple-choice question for testing assistant's distortion perception capabilities.\n"
            + " #### Requirements for Questions and Answers\n"
            + " 1. **Clarity and Focus**: Ensure questions are clear, straightforward, and directly related to the position of the distortion. Do not be too specific but should be general enough.\n"
            + " 2. **Conciseness and Core Information**: Answers should be concise and contain only the essential information, focusing on the core details with minimal words.\n"
            + " 3. **False Candidates**: Provide several false answers under the key 'false candidates' for each question. These must:\n"
            + " - Appear reasonable based on the question."
            + " - Share the same format and subject as the correct answer."
            + " - Contradict the provided image and distortion information."
            + " - Be of a similar **information content** to the correct answer, ensuring balance and avoiding misdirection "
            + " 4. **Avoid Redundancy**: Do not generate question, answer and false candidates pairs that address the same distortion and its locations.\n"
            + " 5. **Information Usage**: Fully utilize the provided image and distortion information to generate diverse pairs.\n"
            + " 6. **Logical Consistency**: Ensure all questions and answers are grammatically correct, logically consistent, and free from contradictions.\n"
            + " - Include misleading elements by mixing characteristics from other distortions (if any), challenging the assistant to identify the specific distortion accurately."
            + " 7. **Question Number**: Limit the total number of pairs to less than 8.\n"
            + " 8. **Quality over Quantity**: Prioritize generating high-quality, insightful question, answer, false candidates pairs. They do not need to match the provided information exactly but should extract and synthesize the key points from the provided information. Avoid questions that can be answered purely from the description of the answer and false candidates without needing to analyze the image."
        )


        try:
            print("Generating question answer pairs...")
            content= gpt4o(query, system_prompt, example_meta_inputs, example_gpt_output_strs)
            print(content)
            resp_dict = json.loads(content)
            meta_item["position"] = resp_dict["question_answer_pair"]
            position_data.append(meta_item)
            print(f"Position question for {img_name} generated.")
            with open(position_file, "w") as fw:
                json.dump(position_data, fw, indent=4, ensure_ascii=False)
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
    fail_dir = os.path.dirname(position_file)
    os.makedirs(fail_dir, exist_ok=True)
    fail_path = os.path.join(fail_dir, "res_fail.txt")
    with open(fail_path, "w") as fw:
        fw.write("\n".join(dist_paths_error))
