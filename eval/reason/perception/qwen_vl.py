from prompt import process_qbench
from doi_prompt import process_benchmark_mcq, process_benchmark_saq
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
parser.add_argument('--device', type=str, default='cuda', help='device to run the model')
parser.add_argument('--save_path', type=str, required=True, help='path to save the predicted answers')
parser.add_argument('--max_new_tokens', default=512, type=int, help='max number of new tokens to generate')
parser.add_argument('--eval_dataset', type=str, required=True, choices=['q-bench', 'doi-bench-mcq', 'doi-bench-saq'], help='datasets to evaluate')
args = parser.parse_args()
if args.eval_dataset == 'q-bench':
    raw_data, processed_data = process_qbench()
elif args.eval_dataset == 'doi-bench':
    raw_data, processed_data = process_benchmark_mcq()
elif args.eval_dataset == 'doi-bench-saq':
    raw_data, processed_data = process_benchmark_saq()

tokenizer = AutoTokenizer.from_pretrained(args.model_base, trust_remote_code=True)
if os.path.exists(args.save_path):
    print(f"File {args.save_path} already exists. Exiting...")
    exit()
# use bf16
# model_path = "../models/qb_finetuen_weights/qwen-vl-chat_lora-True_qlora-False-qinstruct_qalign"
model_path = args.model_path
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map=args.device, 
    trust_remote_code=True, 
    fp16=True
).eval()
print(model.dtype)

system = "You are an expert in image quality assessment."
for gt, data in tqdm(zip(raw_data,processed_data), total=len(raw_data)):
    # Preparation for inference
    prompt = [
        {'image': data['content'][0]['image']},
        {'text': data['content'][1]['text']},
    ]

    query = tokenizer.from_list_format(prompt)
    response, history = model.chat(tokenizer, query=query, system=system, history=None, max_new_tokens=args.max_new_tokens, do_sample=False, top_p=None, top_k=None, temperature=None)
    print(query)
    gt["pred_ans"] = response
    try:
        print("GT:", gt["correct_ans"])
        print("Pred:", gt["pred_ans"])
    except:
        print("GT:", gt["ground_truth"])
        print("Pred:", gt["pred_ans"])

# Save the predicted answers to a file
with open(args.save_path, 'w') as f:
    json.dump(raw_data, f, indent=4, ensure_ascii=False)