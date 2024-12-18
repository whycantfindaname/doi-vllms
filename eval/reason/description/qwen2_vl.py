from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import re
import json
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models/qwen2-vl-7b-instruct', help='path to the model')
parser.add_argument('--model_base', type=str, default='../models/qwen2-vl-7b-instruct', help='base name of the model')
parser.add_argument('--device', type=str, default='cuda:0', help='device to run the model')
parser.add_argument('--eval_file', type=str, default='../datasets/val_json/q_pathway_eval.json', help='path to the evaluation file')
parser.add_argument('--image_folder', type=str, default='../datasets/images', help='path to the folder of images')
parser.add_argument('--save_path', type=str, default='results/gvlmiqa_bench/qwen2vl/qwen2vl_description_gvlmiqabench.json', required=True, help='path to save the predicted answers')
parser.add_argument('--max_new_tokens', default=1024, type=int, help='max number of new tokens to generate')
parser.add_argument('--query', default='Describe and evaluate the quality of the image.', type=str)
args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_base, trust_remote_code=True)
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
input()
# default: Load the model on the available device(s)
model_path = args.model_path
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype="auto", 
    device_map=args.device,
    attn_implementation="flash_attention_2"
)
print(model.hf_device_map)
print(model.dtype)  

try:
    processor = AutoProcessor.from_pretrained(args.model_path)
except:
    processor = AutoProcessor.from_pretrained(args.model_base)

# prompt1 = "Describe and evaluate the quality of the image."
# prompt2 = "Evaluate the quality of the image and provide a comprehensive explanation."

for image in tqdm(not_complete, total=len(not_complete)):
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": os.path.join(image_folder, image),
            },
            {"type": "text", "text": args.query},
        ],
    }
]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device).to(model.dtype)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(image)
    print(output_text[0])
    save_data.append({'image': os.path.join(image_folder, image), 'answer': output_text[0]})
    # Save the predicted answers to a file
    with open(args.save_path, 'w') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)