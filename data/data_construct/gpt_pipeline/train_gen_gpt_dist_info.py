import argparse
import base64
import json
import os

# import weave
import openai
from PIL import Image
import random
from openai import OpenAI
from collections import OrderedDict
from data_utils import load_json, convert_train_dist_info, encode_img


OPENAI_API_KEY = "sk-tE7K8vJ9Dla5zDMx87F9EeB7372340C68067179938991e54"
OPENAI_API_BASE = "https://api.gpt.ge/v1/"
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

parser = argparse.ArgumentParser(
    description="To Prompt GPT-4 for Image Quality Assessment"
)
parser.add_argument("--meta_file", type=str, default='data/meta_json/train-v1/test/dist_info_v2_clean.json')
parser.add_argument("--image_folder", type=str, default='../datasets/images/train_vis_dist')
parser.add_argument("--desp_file", type=str, default='data/meta_json/description.json')
parser.add_argument("--save_file", type=str, default='data/meta_json/train-v1/test/test_dist_info_v2.json')

system_prompt = (
    "You are an expert in image quality assessment. You are given two images: the first image is the original, and the second image is the same with bounding boxes and an ID in the upper-left corner of each box. The bounding boxes mark the regions affected by distortion. "
    + "Each distortion is described in terms of its bounding box ID and distortion name." #, and coordinates. The bounding box coordinates are in the format [tl_x, tl_y, br_x, br_y], where 'tl' is the top-left corner and 'br' is the bottom-right corner. The coordinates are normalized within the range of 0 to 1000.\n"
    + "Please provide an analysis for each bounding box and its corresponding distortion in the following format:\n"
    + "bbox i:\n"
    + " 1. **Position**: Identify which objects in the first image are affected by the distortion, based on the the region marked by the bounding box in the second image. The objects' position should be specific, combining notable features or locations within the bounding box and its position in the image. For example, such as 'the tree branches in the upper-left background' or 'The clothing on the shoulder of the young boy riding the mechanical horse' . If multiple objects are affected, list them all.\n"
    + " 2. **Severity**: Assess the severity of the distortion, which can be one of the following three levels:\n"
    + "   - **Minor**: Barely perceptible upon close inspection.\n"
    + "   - **Moderate**: Clearly noticeable at a glance.\n"  
    + "   - **Severe**: Significantly impacts the overall perception of the image.\n"  
    + " 3. **Visual Manifestations**: Describe how the distortion appears in the affected areas. Include any changes in the shape or appearance of objects. For example, with motion blur, affected objects may appear with stretched trajectories or blurred details.\n"
    + " 4. **Perception Impact**: Evaluate how the distortion affects the visual perception of the objects. Focus on low-level attributes like visibility, sharpness, detail, color, clarity, lighting, texture, and composition. Use the exact name of the distortion as provided, without any modifications.\n"
    + "Use the provided images and distortion information solely as a reference to guide your analysis. Do not directly quote or replicate the images or distortion details in your response."
)


# Weave will track the inputs, outputs and code of this function
# @weave.op()
def gpt4o(origin_img_path, distorted_img_path, query, system_prompt):
    mime_type, img_base64 = encode_img(origin_img_path)
    mime_type_1, img_base64_1 = encode_img(distorted_img_path)
    resp_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "analysis_response_schema",
            "strict": True,
            "schema": {
            "type": "object",
            "properties": {
                "distortions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "description": "The id of the bbox. Should be a only number."
                        },
                        "position": {
                            "type": "string",
                            "description": "Only the names of the objects in the image affected by the distortion."
                        },
                        "severity": {
                            "type": "string",
                            "description": "One of the following levels: minor, moderate, severe."
                        },
                        "visual manifestation": {
                            "type": "string",
                            "description": "Description of the visual manifestation of the distortion in the image."
                        },
                        "perception impact": {
                            "type": "string",
                            "description": "Evaluation of the impact of the distortion on the visual perception of the affected objects."
                        },
                    },
                    "required": ["id", "position", "severity", "visual manifestation", "perception impact"],
                    "additionalProperties": False
                }
                }
            },
                "required": ["distortions"],
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
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{img_base64}"
                            },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type_1};base64,{img_base64_1}"
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

if __name__ == "__main__":
    # weave.init("image quality assessment")
    args = parser.parse_args()

    meta_file = args.meta_file
    image_folder = args.image_folder
    save_file = args.save_file  
    desp_file = args.desp_file
    meta_data = load_json(meta_file)
    if os.path.exists(desp_file):
        desp_data = load_json(desp_file)
    else:
        print("Please generate description first")
    if os.path.exists(save_file):
        save_data = load_json(save_file)
    else:
        save_data = meta_data.copy()

    dist_paths_error = []

    idx_meta_start = 0
    idx_meta_end = -1
    
    for idx_meta, meta_item in enumerate(save_data[idx_meta_start:]):
        if "assessment" in meta_item:
            meta_item.pop("assessment", None)
        img_name = meta_item["image"]
        img_path = os.path.join('../datasets/images/doi-images-all', img_name)
        print("=" * 100)
        print(f"Processing {img_name}...")
        desp_item = next(
            (item for item in desp_data if item["filename"] == img_name), None
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
        if isinstance(meta_item['distortions'], str):
            continue

        distortions = meta_item['distortions']
        # Step 1: Group distortions by type
        grouped_distortions = {}
        for dist_item in distortions:
            distortion_type = dist_item['distortion']
            if distortion_type not in grouped_distortions:
                grouped_distortions[distortion_type] = []
            grouped_distortions[distortion_type].append(dist_item)

        # Step 2: Process each distortion type separately
        for distortion_type, items in grouped_distortions.items():
            # Prepare distortion information for GPT
            dist_info = f"[Image caption]\n{description[0]}\n[Distortion Information]\n"
            if 'position' in items[0].keys():
                continue
            for i, dist_item in enumerate(items):
                coord = dist_item['coordinates']
                id = dist_item['id']
                dist_info += f"{id}: The distortion is {distortion_type.lower()}.\n"
            
            # Print distortion type and information for debugging
            print("=" * 50)
            print(f"Processing distortion type: {distortion_type}")
            print(dist_info)
            input()
            
            # Generate analysis using GPT for the current distortion type
            try:
                dist_image_folder = os.path.join(image_folder, img_name.split('.')[0])
                distortion_type_name = distortion_type.replace(" ", "_")
                dist_image= os.path.join(dist_image_folder, f'{distortion_type_name}.jpg')
                print(f"Generating distortion analysis for {distortion_type} of {dist_image}")
                content = gpt4o(img_path, dist_image, dist_info, system_prompt)
                print(content)
                
                # Parse the response from GPT
                resp_dict = json.loads(content)
                resp_dist = resp_dict['distortions']
                
                # Ensure the lengths match
                assert len(resp_dist) == len(items), (
                    f"Mismatch between response ({len(resp_dist)}) and input distortions ({len(items)}) "
                    f"for distortion type: {distortion_type}"
                )
                
                # Merge GPT response into the original data structure
                for i, resp in enumerate(resp_dist):
                    items[i]['position'] = resp['position']
                    items[i]['severity'] = resp['severity']
                    items[i]['perception impact'] = resp['perception impact']
                    items[i]['visual manifestation'] = resp['visual manifestation']
                print(items)
                input()
                with open(save_file, "w") as fw:
                    json.dump(save_data, fw, indent=4, ensure_ascii=False)
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
                dist_paths_error.append(img_name)
    fail_path = os.path.join(os.path.dirname(save_file), "res_fail.txt")
    with open(fail_path, "w") as fw:
        fw.write("\n".join(dist_paths_error))
