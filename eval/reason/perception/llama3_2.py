from transformers import MllamaForConditionalGeneration, AutoProcessor
from prompt import process_qbench
from tqdm import tqdm
import torch
import json
from PIL import Image
import os
import argparse
raw_data, processed_data = process_qbench()
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models/Llama-3.2-11B-Vision-Instruct', help='path to the model')
parser.add_argument('--model_base', type=str, default='../models/Llama-3.2-11B-Vision-Instruct', help='base name of the model')
parser.add_argument('--device', type=str, default='auto', help='device to run the model')
parser.add_argument('--save_path', type=str, required=True, help='path to save the predicted answers')
args = parser.parse_args()
# if os.path.exists(args.save_path):
#     print(f"File {args.save_path} already exists. Exiting...")
#     exit()
model_path = args.model_path
# default: Load the model on the available device(s)
model = MllamaForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    device_map=args.device,
)
print(model.hf_device_map)
print(model.dtype)  

processor = AutoProcessor.from_pretrained(args.model_base)


for gt, data in tqdm(zip(raw_data,processed_data), total=len(raw_data)):
    image_file = data['content'][0]["image"]
    data['content'][0] =  {"type": "image"}
    conv = [data]
    # Preparation for inference
    raw_image = Image.open(image_file).convert("RGB")
    input_text = processor.apply_chat_template(conv, add_generation_prompt=True)
    print(input_text)
    inputs = processor(raw_image, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    # print(generated_ids_trimmed)
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    gt["pred_ans"] = output_text
    print(gt["correct_ans"])
    print(gt["pred_ans"])
    # input()

# Save the predicted answers to a file
save_path = args.save_path
os.makedirs(os.path.dirname(save_path), exist_ok=True)
with open(save_path, 'w') as f:
    json.dump(raw_data, f, indent=4, ensure_ascii=False)