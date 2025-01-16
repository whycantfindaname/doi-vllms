from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from gvlmiqa_prompt import process_benchmark_ground
from tqdm import tqdm
import json
raw_data, processed_data = process_benchmark_ground()
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
parser.add_argument('--save_path', type=str, required=True, help='path to save the predicted answers')
args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.model_base, trust_remote_code=True)
if os.path.exists(args.save_path):
    print(f"File {args.save_path} already exists. Exiting...")
    exit()
# use bf16
model_path = args.model_path
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map=args.device, 
    trust_remote_code=True, 
    bf16=True
).eval()
print(model.dtype)


for data in tqdm(processed_data, total=len(processed_data)):
    # Preparation for inference
    system = "You are an expert in image quality assessment."
    prompt = [
        {'image': data['content'][0]['image']},
        {'text': data['content'][1]['text']},
    ]
    qtype = data['type']
    print(qtype)

    query = tokenizer.from_list_format(prompt)
    response, history = model.chat(tokenizer, query=query, system=system, history=None)
    print(query)
    print(response)
    for item in raw_data:
        if item["image"] == os.path.basename(data['content'][0]['image']):
            if qtype == "vis":
                item["pred_vis"] = response
            else:
                item["pred_cap"] = response
# Save the predicted answers to a file
with open(args.save_path, 'w') as f:
    json.dump(raw_data, f, indent=4, ensure_ascii=False)