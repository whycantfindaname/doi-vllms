from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from qwen_vl_utils import process_vision_info
from prompt import process_qbench
from doi_prompt import process_benchmark_mcq, process_benchmark_saq
from tqdm import tqdm
import torch
import json
from PIL import Image
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models/llava-onevision-qwen2-7b-ov-hf', help='path to the model')
parser.add_argument('--model_base', type=str, default='../models/llava-onevision-qwen2-7b-ov-hf', help='base name of the model')
parser.add_argument('--device', type=str, default='cuda', help='device to run the model')
parser.add_argument('--save_path', type=str, required=True, help='path to save the predicted answers')
parser.add_argument('--eval_dataset', type=str, required=True, choices=['q-bench', 'doi-bench-mcq', 'doi-bench-saq'], help='datasets to evaluate')
args = parser.parse_args()
if args.eval_dataset == 'q-bench':
    raw_data, processed_data = process_qbench()
elif args.eval_dataset == 'doi-bench-mcq':
    raw_data, processed_data = process_benchmark_mcq('data/meta_json/benchmark-v1/release/mcq_data.json')
elif args.eval_dataset == 'doi-bench-saq':
    raw_data, processed_data = process_benchmark_saq()
if os.path.exists(args.save_path):
    print(f"File {args.save_path} already exists. Exiting...")
    exit()
else:
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
model_path = args.model_path
# default: Load the model on the available device(s)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True,
    device_map=args.device, 
    attn_implementation="flash_attention_2"
)
print(model.hf_device_map)
print(model.dtype)  

processor = AutoProcessor.from_pretrained(args.model_base)


for gt, data in tqdm(zip(raw_data,processed_data), total=len(raw_data)):
    image_file = data['content'][0]["image"]
    data['content'][0] =  {"type": "image"}
    conv = [data]
    # Preparation for inference
    prompt = processor.apply_chat_template(
        conv,  add_generation_prompt=True
    )
    print(prompt)
    raw_image = Image.open(image_file).convert("RGB")

    inputs = processor(images=raw_image, text=prompt, return_tensors='pt')
    inputs = inputs.to(model.device).to(model.dtype)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False, top_p=None, top_k=None, temperature=None)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = generated_text.strip().split("\n")[-1]
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
