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
parser.add_argument("--meta_file", type=str, default='data/meta_json/train/release/train_10k_seagull_clean.json')
parser.add_argument("--image_folder", type=str, default='../datasets/images/gvlmiqa_train/')
parser.add_argument("--save_file", type=str, default='data/meta_json/train/test/test_dist_info_v1.json')


# Weave will track the inputs, outputs and code of this function
# @weave.op()
def gpt4o(img_path, query, system_prompt):
    mime_type, img_base64 = encode_img(img_path)
    print("Encoded image data length:", len(img_base64))  # 添加调试信息，打印前30个字符
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
                            "type": "string",
                            "description": "The id of the bbox."
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
                            },
                    },
                ],
            }
        ],
        temperature=0.8,
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

    meta_data = load_json(meta_file)
    if os.path.exists(save_file):
        save_data = load_json(save_file)
    else:
        save_data = []

    dist_paths_error = []

    idx_meta_start = 0
    idx_meta_end = -1
    
    for idx_meta, meta_item in enumerate(meta_data[idx_meta_start:]):
        img_name = meta_item["image"]
        if img_name in [item["image"] for item in save_data]:
            print(f"{img_name} has been generated, skip.")
            continue
        print("=" * 100)
        print(idx_meta + idx_meta_start)
        print(img_name)
        img_path = os.path.join(image_folder, img_name)
        try:
            global_dist_class = meta_item["global"]
            local_dist_class = meta_item["local"]
            dist_class = {**global_dist_class, **local_dist_class}
            dist_info = convert_train_dist_info(dist_class)
        except:
            continue
            
        # print(dist_info)
        # input()        

        if dist_class != {}:
            system_prompt =(
                "  You are an image quality assessment expert."
                + " You are given an image along with distortion information for that image."
                + " The distortion information is provided in the form of a dictionary:\n"
                + " dict(bbox i: {'distortion': distortion name , 'coordinates':{'tl':{x, y}, 'br':{x, y}}}),"
                + " where i represents the i-th bbox, tl is the top-left corner, and br is the bottom-right corner."
                + " The coordinates are normalized within the range of 0 to 1000.\n"
                + " Please return an analysis for each bbox and its corresponding distortion in the following format:\n"
                + " bbox i:\n"
                + " 1.Analyze which objects in the image are affected by the distortion based on the bounding box coordinates and the content of the image. Be as specific as possible when identifying the objects, including references to any notable features or locations in the image, such as 'the silhouette of the person near the center' or 'the foliage in the upper-left corner'. If multiple objects are affected, list them all.\n"
                + " 2.Analyze the visual manifestation of the distortion in the image. Describe how the distortion appears in the affected areas, including any noticeable changes in the shape, texture, or clarity of the objects. For example, for motion blur, the affected objects might appear to have stretched trajectories with trailing edges or blurred details. Look for unique effects and nuances in different regions.\n"
                + " 3.Assess the severity of the distortion, and respond with one of the following levels: minor, moderate, severe.\n"
                + " 4.Evaluate the impact of the distortion on the visual perception of the affected objects. Focus on how the distortion affects the low-level attributes such as visibility, sharpness, detail, color, clarity, lighting, texture, and composition. Avoid a rigid or repetitive response and make sure to highlight the specific effect the distortion has on the objects mentioned in step 1, explaining in detail how their appearance is impacted by the distortion."
                + " When describing distortions, use the exact provided names without any modifications."
                + " Avoid repeating the same pattern; instead, look at how each distortion affects the perception of different objects or areas in various ways."
            )

        else:
            print("No distortion information found for this image.")
            save_data.append({"image": img_name, "distortions": "There is no distortion in this image."})
            with open(save_file, "w") as fw:
                json.dump(save_data, fw, indent=4, ensure_ascii=False)
            continue

        try:
            print("Generating distortion analysis...")
            content = gpt4o(img_path, dist_info, system_prompt)
            print(type(content))
            resp_dict = json.loads(content)
            resp_dict["image"] = img_name
            resp_dict['distortion_info'] = dist_info
            ordered_resp_dict = OrderedDict([
                ("image", resp_dict["image"]),
                ('distortion_info', resp_dict['distortion_info']),
                ('distortions', resp_dict['distortions'])
            ])
            print(ordered_resp_dict)
            save_data.append(ordered_resp_dict)
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
