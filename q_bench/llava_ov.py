from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from qwen_vl_utils import process_vision_info
from prompt import process_qbench
from tqdm import tqdm
import torch
import json
from PIL import Image
raw_data, processed_data = process_qbench()

# default: Load the model on the available device(s)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "../models/llava-onevision-qwen2-7b-ov-hf", 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True,
    device_map="auto", 
    attn_implementation="flash_attention_2"
)
print(model.hf_device_map)
print(model.dtype)  

processor = AutoProcessor.from_pretrained('../models/llava-onevision-qwen2-7b-ov-hf')


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
with open('results/llava_ov_qbench.json', 'w') as f:
    json.dump(raw_data, f, indent=4, ensure_ascii=False)