from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from prompt import process_qbench
# from gvlmiqa_prompt import process_benchmark
from tqdm import tqdm
import re
import json
raw_data, processed_data = process_qbench()
# raw_data, processed_data = process_benchmark()
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models/qwen2-vl-7b-instruct', help='path to the model')
parser.add_argument('--model_base', type=str, default='../models/qwen2-vl-7b-instruct', help='base name of the model')
parser.add_argument('--device', type=str, default='cuda:0', help='device to run the model')
parser.add_argument('--save_path', type=str, required=True, help='path to save the predicted answers')
parser.add_argument('--skip', type=str, default='yes')
args = parser.parse_args()

if os.path.exists(args.save_path) and args.skip.lower() == 'yes':
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
min_pixels = 4*28*28
max_pixels = 3072*28*28
processor = AutoProcessor.from_pretrained(args.model_base)
# processor = AutoProcessor.from_pretrained(args.model_base, min_pixels=min_pixels, max_pixels=max_pixels)


for gt, data in tqdm(zip(raw_data,processed_data), total=len(raw_data)):
    data['content'][0]['min_pixels'] = min_pixels
    data['content'][0]['max_pixels'] = max_pixels
    messages = [data]
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
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Attempt to parse output_text[0] as JSON if it's a string
    if isinstance(output_text, list) and len(output_text) > 0 and isinstance(output_text[0], str):
        try:
            # Replace single quotes with double quotes
            json_str = re.sub(r"(?<!\\)'", '"', output_text[0].strip())
            
            # Parse the modified string as JSON
            parsed_output = json.loads(json_str)
            
            if isinstance(parsed_output, list) and len(parsed_output) > 0 and isinstance(parsed_output[0], dict):
                gt["pred_ans"] = parsed_output[0].get('text', '')  # Safely get 'text' if it exists
            else:
                print("Parsed output has an unexpected structure:", parsed_output)
                gt["pred_ans"] = ""  # Assign a default or handle as needed
        except json.JSONDecodeError:
            print("Failed to parse output_text as JSON after replacements:", json_str)
            gt["pred_ans"] = json_str[0]  # Handle the case where parsing fails
    else:
        print("Unexpected structure for output_text:", output_text)
        gt["pred_ans"] = output_text[0]  # Handle cases where output_text is not as expected

    print(gt["correct_ans"])
    print(gt["pred_ans"])

# Save the predicted answers to a file
with open(args.save_path, 'w') as f:
    json.dump(raw_data, f, indent=4, ensure_ascii=False)