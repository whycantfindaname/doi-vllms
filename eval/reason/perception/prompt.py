import json
import os
def convert_to_mcq(data):
    # Extract relevant information from the input dictionary
    question = data["question"]
    candidates = data["candidates"]
    correct_ans = data["correct_ans"]  # This is the actual answer text
    
    # Map the candidates to letters A, B, C, ...
    options = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(candidates)])
    
    # Find the index of the correct answer
    if correct_ans in candidates:
        correct_ans_index = candidates.index(correct_ans)
        correct_ans_letter = chr(65 + correct_ans_index)
    else:
        raise ValueError(f"Correct answer '{correct_ans}' not found in candidates.")
    
    # Format the question and options in the desired output format
    formatted_output = f"{question}\n{options}\nAnswer with the option's letter from the given choices directly."
    
    return formatted_output, correct_ans_letter

def process_qbench(image_folder="../datasets/images/llvisionqa_qbench_dev/dev", json_file='../datasets/q-bench/llvisionqa_dev.json'):
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    processed_data = []
    for item in raw_data:
        image_path = os.path.join(image_folder, item["img_path"])
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist. Skipping...")
            continue
        text, _ = convert_to_mcq(item)
        processed_item = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text},
            ],
        }
        processed_data.append(processed_item)
    return raw_data, processed_data

def process_qbench_eval(image_folder="../datasets/images/llvisionqa_qbench_dev/dev", json_file='../datasets/q-bench/llvisionqa_dev.json'):
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    processed_data = []
    for item in raw_data:
        image_path = os.path.join(image_folder, item["img_path"])
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist. Skipping...")
            continue
        text, correct_ans_letter = convert_to_mcq(item)
        conv = [{"from": "human", "value": f'<image>{text}'}, {"from": "gpt", "value": f"{correct_ans_letter}."}]
        processed_item = {
            "image": image_path,
            "conversations": conv
        }
        processed_data.append(processed_item)
    return processed_data

if __name__ == '__main__':

    raw_data, processed_data = process_qbench()
    print(raw_data[1])
    print(processed_data[1])
    # processed_data = process_qbench_eval()
    # print(processed_data[1])
    # with open('../datasets/q-bench/q_bench_eval.json', 'w') as f:
    #     json.dump(processed_data, f, indent=4, ensure_ascii=False)