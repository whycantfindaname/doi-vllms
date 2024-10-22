from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from prompt import process_qbench
from tqdm import tqdm
import json
raw_data, processed_data = process_qbench()

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)


tokenizer = AutoTokenizer.from_pretrained("../models/Qwen-VL-Chat", trust_remote_code=True)

# use bf16
model = AutoModelForCausalLM.from_pretrained(
    "../models/Qwen-VL-Chat", 
    device_map="auto", 
    trust_remote_code=True, 
    bf16=True
).eval()


for gt, data in tqdm(zip(raw_data,processed_data), total=len(raw_data)):
    # Preparation for inference
    prompt = [
        {'image': data['content'][0]['image']},
        {'text': data['content'][1]['text']},
    ]

    query = tokenizer.from_list_format(prompt)
    response, history = model.chat(tokenizer, query=query, history=None)
    print(query)
    gt["pred_ans"] = response
    print(gt["correct_ans"])
    print(gt["pred_ans"])

# Save the predicted answers to a file
with open('results/qwen_vl_qbench.json', 'w') as f:
    json.dump(raw_data, f, indent=4, ensure_ascii=False)