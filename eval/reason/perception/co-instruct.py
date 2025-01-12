import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt import process_qbench
from doi_prompt import process_benchmark_mcq, process_benchmark_saq
from tqdm import tqdm
import json
raw_data, processed_data = process_qbench()
save_path = 'results/q_bench/mplug-owl2/co-instruct.json'
raw_data, processed_data = process_benchmark_mcq()
save_path = 'results/doi_bench/mplug-owl2/co-instruct.json'
raw_data, processed_data = process_benchmark_saq()
save_path = 'results/doi_bench/mplug-owl2/co-instruct_position.json'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

model = AutoModelForCausalLM.from_pretrained("../models/co-instruct", 
                                             trust_remote_code=True, 
                                             torch_dtype=torch.float16,
                                             attn_implementation="eager", 
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("../models/co-instruct", trust_remote_code=True)
from PIL import Image
# prompt = "USER: The image: <|image|> Why is the overall quality of the image is not good? ASSISTANT:"
# url = "https://raw.githubusercontent.com/Q-Future/Q-Align/main/fig/singapore_flyer.jpg"
# image = Image.open(requests.get(url,stream=True).raw)
# model.chat(prompt, [image], max_new_tokens=200)

for gt, data in tqdm(zip(raw_data,processed_data), total=len(raw_data)):
    # Preparation for inference
    image_path = data['content'][0]['image']
    query = data['content'][1]['text']
    print(query)
    image = Image.open(image_path).convert('RGB')
    prompt = "USER: The image: <|image|> " + query + " ASSISTANT:"
    len, generated_ids = model.chat(prompt, [image], max_new_tokens=512)
    generated_ids[generated_ids == -200] = tokenizer.pad_token_id
    print(generated_ids)
    input()
    answer = tokenizer.batch_decode(generated_ids[:, len:], skip_special_tokens=True)[0]
    gt["pred_ans"] = answer
    try:
        print("GT:", gt["correct_ans"])
        print("Pred:", gt["pred_ans"])
    except:
        print("GT:", gt["ground_truth"])
        print("Pred:", gt["pred_ans"])
    # Save the predicted answers to a file
    with open(save_path, 'w') as f:
        json.dump(raw_data, f, indent=4, ensure_ascii=False)
