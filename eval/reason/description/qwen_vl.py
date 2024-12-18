from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models/Qwen-VL-Chat', help='path to the model')
parser.add_argument('--model_base', type=str, default='../models/Qwen-VL-Chat', help='base name of the model')
parser.add_argument('--device', type=str, default='cuda:0', help='device to run the model')
parser.add_argument('--eval_file', type=str, default="eval/benchmark_2k.json", help='path to the evaluation file')
parser.add_argument('--image_folder', type=str, default='../datasets/image/gvlmiqa_bench', help='path to the folder of images')
parser.add_argument('--save_path', type=str, default='results/gvlmiqa_bench/qwenvl/qwenvl_description_gvlmiqabench.json', required=True, help='path to save the predicted answers')
parser.add_argument('--max_new_tokens', default=1024, type=int, help='max number of new tokens to generate')
parser.add_argument('--query', default='Describe and evaluate the quality of the image.', type=str)
args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_base, trust_remote_code=True)
with open(args.eval_file,'r') as f:
    data = json.load(f)
image_folder = args.image_folder
if os.path.exists(args.save_path):
    save_data = json.load(open(args.save_path, 'r'))
else:
    save_data = []
not_complete = []
for item in data[len(save_data):]:
    try:
        not_complete.append(item['filename'])
    except:
        not_complete.append(item['image'])
print(len(not_complete))
input()
# use bf16
# model_path = "../models/qb_finetuen_weights/qwen-vl-chat_lora-True_qlora-False-qinstruct_qalign"
model_path = args.model_path
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map=args.device, 
    trust_remote_code=True, 
    bf16=True
).eval()
print(model.dtype)

system = "You are an expert in image quality assessment."
# prompt1 = "Describe and evaluate the quality of the image."
# prompt2 = "Evaluate the quality of the image and provide a comprehensive explanation."
for image in tqdm(not_complete , total=len(not_complete)):
    # Preparation for inference
    image_path = os.path.join(image_folder, image)
    prompt = [
        {'image': image_path},
        {'text': args.query},
    ]

    query = tokenizer.from_list_format(prompt)
    response, history = model.chat(tokenizer, query=query, system=system, history=None, max_new_tokens=args.max_new_tokens)
    print(image_path)
    print(response)
    save_data.append({'image': image, 'answer': response})

    # Save the predicted answers to a file
    with open(args.save_path, 'w') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)