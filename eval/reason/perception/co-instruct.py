import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt import process_qbench
# from gvlmiqa_prompt import process_benchmark
from tqdm import tqdm
import json
import requests
raw_data, processed_data = process_qbench()
# raw_data, processed_data = process_benchmark()
save_path = 'results/q_bench/mplug-owl2/co-instruct.json'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

model = AutoModelForCausalLM.from_pretrained("../models/co-instruct", 
                                             trust_remote_code=True, 
                                             torch_dtype=torch.float16,
                                             attn_implementation="eager", 
                                             device_map={"":"cuda:0"})
tokenizer = AutoTokenizer.from_pretrained("../models/co-instruct", trust_remote_code=True)
from PIL import Image
prompt = "USER: The image: <|image|> Why is the overall quality of the image is not good? ASSISTANT:"
url = "https://raw.githubusercontent.com/Q-Future/Q-Align/main/fig/singapore_flyer.jpg"
image = Image.open(requests.get(url,stream=True).raw)
model.chat(prompt, [image], max_new_tokens=200)
# for gt, data in tqdm(zip(raw_data,processed_data), total=len(raw_data)):
#     # Preparation for inference
#     image_path = data['content'][0]['image']
#     query = data['content'][1]['text']
#     print(query)
#     image = Image.open(image_path).convert('RGB')
#     prompt = "USER: The image: <|image|> " + query + " ASSISTANT:"
#     len, generated_ids = model.chat(prompt, [image], max_new_tokens=200)
#     generated_ids[generated_ids == -200] = tokenizer.pad_token_id
#     print(generated_ids)
#     answer = tokenizer.batch_decode(generated_ids[:, len:], skip_special_tokens=True)[0]
#     gt["pred_ans"] = answer
#     print("GT:", gt["correct_ans"])
#     print("Pred:", gt["pred_ans"])
#     # Save the predicted answers to a file
#     with open(save_path, 'w') as f:
#         json.dump(raw_data, f, indent=4, ensure_ascii=False)
