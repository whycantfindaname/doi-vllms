from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from qwen_vl_utils import process_vision_info
from prompt import process_qbench
from doi_prompt import process_benchmark_input
from tqdm import tqdm
import torch
import json
from PIL import Image
import os
import argparse
# raw_data, processed_data = process_qbench()
raw_data, processed_data = process_benchmark_input()
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models/llava-onevision-qwen2-7b-ov-hf', help='path to the model')
parser.add_argument('--model_base', type=str, default='../models/llava-onevision-qwen2-7b-ov-hf', help='base name of the model')
parser.add_argument('--device', type=str, default='cuda:1', help='device to run the model')
parser.add_argument('--save_path', type=str, required=True, help='path to save the predicted answers')
args = parser.parse_args()
if os.path.exists(args.save_path):
    print(f"File {args.save_path} already exists. Exiting...")
    exit()
else:
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
model_path = args.model_path
# default: Load the model on the available device(s)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
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
    generated_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = generated_text.strip().split("\n")[-1]
    gt["pred_ans"] = response
    print(gt["correct_ans"])
    print(gt["pred_ans"])

# Save the predicted answers to a file
with open(args.save_path, 'w') as f:
    json.dump(raw_data, f, indent=4, ensure_ascii=False)
