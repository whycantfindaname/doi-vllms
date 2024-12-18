from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from gvlmiqa_prompt import process_benchmark_ground
from tqdm import tqdm
import re
import json
raw_data, processed_data = process_benchmark_ground()
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models/qwen2-vl-7b-instruct', help='path to the model')
parser.add_argument('--model_base', type=str, default='../models/qwen2-vl-7b-instruct', help='base name of the model')
parser.add_argument('--device', type=str, default='cuda:0', help='device to run the model')
parser.add_argument('--save_path', type=str, required=True, help='path to save the predicted answers')
parser.add_argument('--system_prompt', type=str, default='You are an expert in image quality assessment.')
args = parser.parse_args()

if os.path.exists(args.save_path):
    print(f"File {args.save_path} already exists. Exiting...")
    exit()
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

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
try:
    processor = AutoProcessor.from_pretrained(args.model_path)
    # print(processor)
except:
    processor = AutoProcessor.from_pretrained(args.model_base)

for gt, data in tqdm(zip(raw_data,processed_data), total=len(raw_data)):
    messages = [data]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ).replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>", f"<|im_start|>system\n{args.system_prompt}<|im_end|>")
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device).to(model.dtype)
    qtype = data['type']
    print(qtype)
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    raw_output_text = output_text[0]
    # print(raw_output_text)
    cleaned_text = raw_output_text.split('\n<|im_end|>')[0]
    print(cleaned_text)
    match = re.search(r'["\']text["\']: ["\'](.*?)["\']\}', cleaned_text, re.DOTALL)
    if match:
        # 提取匹配的内容
        response = match.group(1)
        # 替换转义字符 \n 为换行
        response = response.replace("\\n", "\n")
        print(response)
    else:
        response = None
        print("未匹配到内容")
    for item in raw_data:
        if item["image"] == os.path.basename(data['content'][0]['image']):
            if qtype == "vis":
                item["pred_vis"] = response
            else:
                item["pred_cap"] = response

    # Save the predicted answers to a file
    with open(args.save_path, 'w') as f:
        json.dump(raw_data, f, indent=4, ensure_ascii=False)