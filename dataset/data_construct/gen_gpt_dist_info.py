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


api_key="AN2vyAdM2O8xu8PJDFI6DuMPanQl7qsF"
client = openai.AzureOpenAI(
        azure_endpoint="https://search.bytedance.net/gpt/openapi/online/v2/crawl",
        api_version="2023-07-01-preview",
        api_key=api_key
    )

parser = argparse.ArgumentParser(
    description="To Prompt GPT-4 for Image Quality Assessment"
)
parser.add_argument("--meta_file", type=str, default='dataset/meta_json/benchmark/benchmark_2k_normalized_cleaned.json')
parser.add_argument("--image_folder", type=str, default='/mnt/bn/lwj/datasets/gvlmiqa_bench/')
parser.add_argument("--save_file", type=str, default='/mlx_devbox/users/jasonliao.21/repo/15211/doi-vllms/dataset/meta_json/benchmark/benchmark_2k_cleaned_dist_info.json')


def encode_img(img_path):
    ext = os.path.splitext(img_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif ext == ".png":
        mime_type = "image/png"
    elif ext == ".webp":
        mime_type = "image/webp"
    elif ext == ".bmp":
        # 转换成jpg格式后编码
        img = Image.open(img_path)
        img.save(img_path.replace(ext, ".jpg"), "JPEG")
        mime_type = "image/jpeg"
        img_path = img_path.replace(ext, ".jpg")
    else:
        raise ValueError("Unsupported image format")
    print(img_path)
    with open(img_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    return mime_type, img_base64


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
                "bboxes": {
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
                            "description": "Analysis of objects in the image affected by the distortion based on the bbox coordinates and image content."
                        },
                        "severity": {
                            "type": "string",
                            "description": "One of the following levels: trivial, minor, moderate, severe, extreme."
                        },
                        "overall impact": {
                            "type": "string",
                            "description": "Evaluatation of the overall impact on visual perception."
                        },
                    },
                    "required": ["id", "position", "severity", "overall impact"],
                    "additionalProperties": False
                }
                }
            },
                "required": ["bboxes"],
                "additionalProperties": False
            }
        }
    }
    resp = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
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
        temperature=0.5,
        max_tokens=500,
        response_format=resp_format
    )
    print(resp.usage)
    content = resp.choices[0].message.content
    return content

def convert_dist_info(dist_class):
    formatted_bboxes = []
    bbox_counter = 1
    
    # Iterate over the local distortions
    for distortion, bboxes in dist_class.items():
        for bbox in bboxes:
            formatted_bbox = (
                f"bbox {bbox_counter}: {{\"distortion\": {distortion}, \"coordinates\":{{'tl':{{\"x\":{bbox['tl']['x']}, \"y\":{bbox['tl']['y']}}}, "
                f"'br':{{\"x\":{bbox['br']['x']}, \"y\":{bbox['br']['y']}}}}}}}"
            )
            formatted_bboxes.append(formatted_bbox)
            bbox_counter += 1
    
    return "\n".join(formatted_bboxes)

def load_json(file_path):
    """Load a JSON file and return its contents."""
    with open(file_path, "r") as file:
        return json.load(file) 

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

    # description_query
    dist_paths_error = []
    not_complete = []
    for item in meta_data:
        img_name = item["filename"]
        if img_name not in [item["filename"] for item in save_data]:
            not_complete.append(item)
    print(len(not_complete))
    idx_meta_start = len(save_data)
    idx_meta_end = -1
    for idx_meta, meta_item in enumerate(not_complete):
        img_name = meta_item["filename"]
        print("=" * 100)
        print(idx_meta + idx_meta_start)
        print(img_name)
        img_path = os.path.join(image_folder, img_name)

        global_dist_class = meta_item["global"]
        local_dist_class = meta_item["local"]
        dist_class = {**global_dist_class, **local_dist_class}
        dist_info = convert_dist_info(dist_class)
        # print(dist_info)
        # input()
        

        if isinstance(dist_class, dict):
            system_prompt =(
                "  You are an image quality assessment expert."
                + " You are given an image along with distortion information for that image."
                + " The distortion information is provided in the form of a dictionary:\n"
                + " dict(bbox i: {'distortion': distortion name , 'coordinates':{'tl':{x, y}, 'br':{x, y}}}),"
                + " where i represents the i-th bbox, tl is the top-left corner, and br is the bottom-right corner."
                + " The coordinates are normalized within the range of 0 to 1000.\n"
                + " Please return an analysis for each bbox and its corresponding distortion in the following format:\n"
                + " bbox i:\n"
                + " 1.Analyze which objects in the image are affected by the distortion based on the bbox coordinates and image content.\n"
                + " 2.Assess the severity of the distortion, and respond with one of the following levels: trivial, minor, moderate, severe, extreme.\n"
                + " 3.Combine the results from steps 1 and 2 to evaluate the overall impact on visual perception.\n"
                # + " Separate the analysis of different bboxes with a newline (\n)."
            )
        else:
            print("No distortion information found for this image.")
            save_data.append({"filename": img_name, "distortion_info": "No distortion information found for this image."})
            continue

        try:
            print("Generating assessment...")
            content = gpt4o(img_path, dist_info, system_prompt)
            resp_dict = json.loads(content)
            resp_dict['filename'] = img_name
            resp_dict['distortion_info'] = dist_info
            ordered_resp_dict = OrderedDict([
                ('filename', resp_dict['filename']),
                ('distortion_info', resp_dict['distortion_info']),
                ('bboxes', resp_dict['bboxes'])
            ])
            print(ordered_resp_dict)
            input()
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
