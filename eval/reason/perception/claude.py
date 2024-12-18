import json
import os
import requests
import base64
from PIL import Image
from tqdm import tqdm
from prompt import process_qbench
OPENAI_API_KEY = "sk-tE7K8vJ9Dla5zDMx87F9EeB7372340C68067179938991e54"

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

def claude(img_path, query):
    mime_type, img_base64 = encode_img(img_path)
    print("Encoded image data length:", len(img_base64))  # 添加调试信息，打印前30个字符
    url = "https://api.gpt.ge/v1/chat/completions"

    payload = json.dumps({
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": query
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{img_base64}"
                }
                }
            ]
        }
    ],
    "max_tokens": 10,
    "temperature": 0.5,
    "stream": False,
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + OPENAI_API_KEY,
    }
    resp = requests.request("POST", url, data=payload, headers=headers).text
    resp = json.loads(resp)
    print(resp)
    print(resp['usage'])
    content = resp['choices'][0]['message']['content']
    return content

if __name__ == "__main__":
    save_dir = 'results/q_bench/close-source'
    save_name = 'claude-3-5-sonnet-20241022.json'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    raw_data, processed_data = process_qbench()
    count = 0
    dist_paths_error = []
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            save_data = json.load(f)
    else:
        save_data = []
    save_data = [item for item in save_data if 'response' in item.keys()]
    print(len(save_data))
    for gt, data in tqdm(zip(raw_data,processed_data), total=len(raw_data)):
        img_path = data['content'][0]['image']
        if save_data and os.path.basename(img_path) in [d['img_path'] for d in save_data]:
            # print(f"Skip {img_path}")
            continue
        query = data['content'][1]['text']
        print(query)
        try:
            answer = claude(img_path, query)
            print(answer)
            gt['response'] = answer
            save_data.append(gt)
            # count += 1
            # if count % 100 == 0:
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
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
