from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from tqdm import tqdm
import torch
import json
from PIL import Image
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models/llava-onevision-qwen2-7b-ov-hf', help='path to the model')
parser.add_argument('--model_base', type=str, default='../models/llava-onevision-qwen2-7b-ov-hf', help='base name of the model')
parser.add_argument('--device', type=str, default='cuda:0', help='device to run the model')
parser.add_argument('--eval_file', type=str, default="eval/benchmark_2k.json", help='path to the evaluation file')
parser.add_argument('--image_folder', type=str, default='../datasets/image/gvlmiqa_bench', help='path to the folder of images')
parser.add_argument('--save_path', type=str, default='eval/reason/llava/llavaov_description_gvlmiqabench.json', required=True, help='path to save the predicted answers')
parser.add_argument('--max_new_tokens', default=1024, type=int, help='max number of new tokens to generate')
parser.add_argument('--query', default='Describe and evaluate the quality of the image.', type=str)
args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
# Load the data
with open(args.eval_file, 'r') as f:
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
# not_complete = data[len(save_data):]

# Load the model and the processor
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


for image in tqdm(not_complete, total=len(not_complete)):
    image_file = os.path.join(image_folder, image)
    conv = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": args.query},
            {"type": "image"},
            ],
        },
    ]
    # Preparation for inference
    prompt = processor.apply_chat_template(
        conv,  add_generation_prompt=True
    )
    print(prompt)
    raw_image = Image.open(image_file).convert("RGB")

    inputs = processor(images=raw_image, text=prompt, return_tensors='pt')
    inputs = inputs.to(model.device).to(model.dtype)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = generated_text.strip().split("\n")[-1]
    print(response)
    item['llava_ov_response'] = response
    save_data.append({'image': image, 'answer': response})

    # Save the predicted answers to a file
    with open(args.save_path, 'w') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)
