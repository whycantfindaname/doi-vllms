import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models/co-instruct', help='path to the model')
parser.add_argument('--model_base', type=str, default='../models/co-instruct', help='base name of the model')
parser.add_argument('--device', type=str, default='cuda:0', help='device to run the model')
parser.add_argument('--eval_file', type=str, default="eval/benchmark_2k.json", help='path to the evaluation file')
parser.add_argument('--image_folder', type=str, default='../datasets/image/gvlmiqa_bench', help='path to the folder of images')
parser.add_argument('--save_path', type=str, default='results/gvlmiqa_bench/mplug-owl2/co-instruct_description.json', required=True, help='path to save the predicted answers')
parser.add_argument('--max_new_tokens', default=1024, type=int, help='max number of new tokens to generate')
parser.add_argument('--query', default='Describe and evaluate the quality of the image.', type=str)
args = parser.parse_args()
save_path = args.save_path
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# file = '../datasets/val_json/q_pathway_eval.json
file = args.eval_file

# 读取JSON文件
with open(file, "r", encoding="utf-8") as file:
    data = json.load(file)
img_list = []
# image_dir = "../datasets/images"
image_dir = args.image_folder
os.makedirs(os.path.dirname(save_path), exist_ok=True)
for i in range(len(data)):
    try:
        image = data[i]["image"]
    except:
        image = data[i]["filename"]
    img_list.append(os.path.join(image_dir, image))
tokenizer = AutoTokenizer.from_pretrained(args.model_path, 
                                            trust_remote_code=True,)
model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                                             trust_remote_code=True, 
                                             torch_dtype=torch.float16,
                                             attn_implementation="eager", 
                                             device_map={"":f"{args.device}"})

from PIL import Image
save_data = []
for i in tqdm(range(len(img_list))):
    # Preparation for inference
    image_path = img_list[i]
    query = args.query
    image = Image.open(image_path).convert('RGB')
    prompt = "USER: The input image: <|image|>. " + query + " ASSISTANT:"
    len, generated_ids = model.chat(prompt, [image], max_new_tokens=args.max_new_tokens)
    generated_ids[generated_ids == -200] = tokenizer.pad_token_id
    answer = tokenizer.batch_decode(generated_ids[:, len:], skip_special_tokens=True)[0]
    print(answer)
    save_data.append({"image": image_path, "answer": answer})

    # Save the predicted answers to a file
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)
