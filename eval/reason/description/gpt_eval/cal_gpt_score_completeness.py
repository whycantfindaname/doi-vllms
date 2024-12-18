import argparse
import base64
import json
import os

import weave
from openai import OpenAI
from PIL import Image
OPENAI_API_KEY = "sk-tE7K8vJ9Dla5zDMx87F9EeB7372340C68067179938991e54"
OPENAI_API_BASE = "https://api.gpt.ge/v1"
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

parser = argparse.ArgumentParser(description="To Prompt GPT-4 for Image Description Evaluation")
parser.add_argument("--golden_file", type=str, default='results/q_pathway/q_pathway_eval.json')
parser.add_argument("--image_folder", type=str, default="../datasets/images")
parser.add_argument("--desp_file", type=str, required=True)

def load_json(file_path):
    with open(file_path, "r") as fr:
        data = json.load(fr)
    return data

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


@weave.op()
def gpt4o(img_path, system_prompt, query):
    mime_type, img_base64 = encode_img(img_path)
    print(f"Encoded image data: {img_base64[:30]}...")  # 添加调试信息，打印前30个字符
    resp = client.chat.completions.create(
        model="gpt-4o",
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
                            "url": f"data:{mime_type};base64,{img_base64}",
                            'detail': 'auto',
                            },
                    },
                ],
            }
        ],
        temperature=0,
        max_tokens=10,
    )
    print(resp.usage)
    content = resp.choices[0].message.content
    return content


if __name__ == "__main__":
    weave.init("image description")
    args = parser.parse_args()
    idx_meta_start = 0
    idx_meta_end = 100

    golden_file = args.golden_file
    desp_file = args.desp_file
    image_folder = args.image_folder
    save_dir = os.path.join(os.path.dirname(desp_file), "completeness")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f"{os.path.basename(desp_file).split('.json')[0]}_completeness_scores.json")
    print("Saving scores to:", save_file)
    # description_query
    dist_paths_error = []
    golden_data = load_json(golden_file)
    desp_data = load_json(desp_file)
    model_name = os.path.basename(desp_file).split(".")[0].split("_description")[0]
    if os.path.exists(save_file):
        save_data = load_json(save_file)
    else:
        save_data = desp_data
        
    for item in save_data:
        if 'scores' in item.keys():
            continue
        else:
            item['scores'] = []

    assert len(golden_data) == len(desp_data)
    num_save_data = len(save_data)
    
    for idx_meta, (golden_item, desp_item) in enumerate(zip(golden_data[idx_meta_start:idx_meta_end], save_data[idx_meta_start:idx_meta_end])):
        print("=" * 100)
        print(idx_meta + idx_meta_start)
        print("Lenght of scores:", len(desp_item['scores']))
        if len(desp_item['scores']) == 5:
            continue
        try:
            image = desp_item['img_A_path']
        except:
            image = desp_item['image']
        assert image == golden_item['image']
        img_path = os.path.join(image_folder, image)
        model_desp = desp_item['answer']
        # Collect all human-generated descriptions (both questions and answers) from all conversations
        golden_desp = ""
        # Iterate over all conversation keys, excluding the 'image' key
        for key, conversation in golden_item.items():
            if key != 'image':  # Skip the 'image' key
                for msg in conversation:
                    if msg['from'] == 'human':
                        # Add the question (if it's from the human)
                        question = msg['value'].replace('\n<image>','')
                        golden_desp += f"Question: {question}\n"
                    elif msg['from'] == 'gpt':
                        # Add the answer (if it's from GPT)
                        golden_desp += f"Answer: {msg['value']}\n"
        
        system_prompt = (
            "You are an expert in image quality assessment. Your task is to evaluate the completeness of the MLLM description "
            + "based on the detailed low-level visual information derived directly from the image and the reference (human-generated) "
            + "question-answer pairs. Focus on whether the MLLM description [MLLM DESC] accurately includes the key low-level visual details "
            + "observed in the image and reflected in the reference question-answer pairs [GOLDEN DESC]. "
            + "Please rate score 2 for completely or almost completely including reference information, 0 for not including at all, "
            + "1 for including part of the information or similar description. "
            + "Please provide only the result in the following format: Score:"
        )

        
        completeness_query = (
            f"[GOLDEN DESC] {golden_desp}\n[MLLM DESC] {model_desp}"
        )

        try:
            assert len(save_data) == num_save_data
            import re
            content = gpt4o(img_path, system_prompt, completeness_query)
            print(f'{model_name}_desp:', model_desp)
            match = re.search(r"Score:\s*(\d+)", content)
            if match:
                # 提取到的数字
                score = int(match.group(1))
                print(score)
            else:
                print("No score found.")
            # Add score only if it's less than 5
            if len(desp_item['scores']) < 5:
                desp_item['scores'].append(score)
            
            # Ensure we don't append the item multiple times
            save_data[idx_meta + idx_meta_start] = desp_item
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
            dist_paths_error.append(img_path)

    fail_path = os.path.join(os.path.dirname(save_file), os.path.basename(save_file).split(".")[0] + "_res_fail.txt")
    with open(fail_path, "w") as fw:
        fw.write("\n".join(dist_paths_error))
