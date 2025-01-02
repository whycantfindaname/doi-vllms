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
parser.add_argument("--save_file", type=str, default='data/meta_json/train/test/test_brief_assess_v1.json')
parser.add_argument("--image_folder", type=str, default='../datasets/images/gvlmiqa_train/')

example_gpt_input = "The image depicts a young boy riding a mechanical horse in an outdoor setting, with another child silhouetted in the foreground.\nThere are three out of foucs blurs in the image. <bbox 1>Moderate out of focus blur</bbox 1> causes the edges of the saddle and the boy's legs to appear soft and lacking in detail. The texture of the saddle is not discernible, and the legs of the boy blend into the saddle, losing definition.<bbox 2>Severe out of focus blur</bbox 2> made the silhouette appear with soft, indistinct edges, blending into the background. The lack of sharpness and definition in this area causes the silhouette to lose its visual impact and presence in the image. The boy's striped shirt and the front part of the horse are also affected by <bbox 3>moderate out of foucs blur</bbox 3>.The stripes on the boy's shirt are indistinct, appearing as blurred lines with little contrast. The front part of the horse lacks definition, with soft edges and a loss of textural detail.\nThere is one <bbox 4>minor edge aliasing effect</bbox 4> affecting tree branches in the background.The edges of the tree branches appear jagged and pixelated, rather than smooth and natural. This creates a distracting effect that detracts from the natural appearance of the landscape.\nThere are two underexposures in the image. <bbox 5>Severe underexposure</bbox 5> causes the right side of the image, including parts of the horse and the background, to appear dark and lacking in detail. The silhouette of the person in the foreground also shows <bbox 6>severe underexposure</bbox 6>. The silhouette is poorly defined, appearing as a dark shape with no visible features. The underexposure causes a loss of detail and texture, making it indistinct and blending it into the background.\nThe <bbox 2>severe out of focus blur</bbox 2> and <bbox 6>severe underexposure</bbox 6> in the foreground have a significant impact on the overall quality. The foreground is important to the overall quality, but it appears blurry and lacks contrast, weakening the sense of visual focus and composition.The overall contrast is weak, with object outlines appearing soft and indistinct. However, no significant noise is observed, and the image maintains a monochromatic appearance without any noticeable color distortions.\nThus, the quality of the image is fair."
example_gpt_output = "- Conclusion: The quality of the image is fair.\n- Quality issues: Blur, Edge aliasing, Underexposure\n    - Out of focus blur: There are three out of focus blurs in the image. A moderate out of focus blur on the edges of the saddle and the boy's legs causes the texture to be unrecognizable, with the legs blending into the saddle, losing detail. A severe out of focus blur in the silhouette in the foreground causes it to appear soft and indistinct, losing visual impact and presence, blending into the background. A moderate out of focus blur affects the boy's striped shirt and the front part of the horse, making the stripes indistinct, appearing as blurred lines with little contrast. The front part of the horse lacks definition, losing textural detail.\n    - Edge aliasing: There is a minor edge aliasing effect affecting tree branches in the background. The edges of the tree branches appear jagged and pixelated, rather than smooth and natural, which creates a distracting effect that detracts from the natural appearance of the landscape.\n    - Underexposure: The right side of the image, including parts of the horse and the background, appears severely underexposed, with a lack of detail. The silhouette of the person in the foreground is also severely underexposed, causing it to appear as a dark shape with no visible features.\n- Scene description: The image depicts a young boy riding a mechanical horse in an outdoor setting, with another child silhouetted in the foreground."

# Weave will track the inputs, outputs and code of this function
@weave.op()
def gpt4o(query, system_prompt):
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
                    {"type": "text", "text": example_gpt_input}
                ],
            },
            {
                "role": "assistant",
                "content": example_gpt_output,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query}
                ],
            }
        ],
        temperature=0.5
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
    save_file = args.save_file
    image_folder = args.image_folder

    meta_data = load_json(meta_file)
    if os.path.exists(save_file):
        save_data = load_json(save_file)
    else:
        save_data = []
    print(len(meta_data))
    print(len(save_data))
    check_flag = check(meta_data, save_data)
    check_duplicate(meta_data)
    dist_paths_error = []
    
    for idx_meta, meta_item in enumerate(meta_data[idx_meta_start:]):
        img_name = meta_item["image"]
        if img_name in [item["image"] for item in save_data]:
            # print(f"{img_name} has been generated, skip.")
            continue
        print("=" * 100)
        print(idx_meta + idx_meta_start)
        print(img_name)
        img_path = os.path.join(image_folder, img_name)

        try:
            overall_assess = clean_text(meta_item['assessment']['overall quality assessment'])  
        except Exception as e:
            print(f"Error occurred while cleaning overall assessment for {img_name}.")
            print(e)
            overall_assess = meta_item['assessment']
        assess_query = f'[Overall Quality Assessment]\n{overall_assess}'
        
        system_prompt_assess_data = (
            "Transform the following detailed image quality assessment into a structured and concise summary. The summary should include:\n"
            + " Conclusion: The quality level of the image (e.g. one of the following: 'bad', 'poor', 'fair', 'good', 'excellent').\n"
            + " Distortion assessment: List and describe the distortions in detailed quality assessment identified, with brief explanations of their impact."
            + " Your analysis for each distortion should start with the number of occurrences of this distortion in the image"
            + " For each distortion, mention the specific effect and how it affects the image."
            + " Use the exact terminology provided for distortion names, do not introduce unrelated terms or deviate from them."
            + " If there is no distortion in the image, summarize the detailed image quality assessment, especially low level attributes (e.g., sharpness, clarity, contrast, lighting).\n"
            + " Scene description: A concise description of the scene in the image, focusing on the key elements."
        )
        
        try:
            print("Generating question answer pairs...")
            content= gpt4o(assess_query, system_prompt_assess_data)
            meta_item["brief_assessment"] = content
            save_data.append(meta_item)
            print(f"Brief assessment for {img_name} generated.")
            with open(save_file, "w") as fw:
                json.dump(save_data, fw, indent=4, ensure_ascii=False)
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
    fail_dir = os.path.dirname(save_file)
    os.makedirs(fail_dir, exist_ok=True)
    fail_path = os.path.join(fail_dir, "res_fail.txt")
    with open(fail_path, "w") as fw:
        fw.write("\n".join(dist_paths_error))
