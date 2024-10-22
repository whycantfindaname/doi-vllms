import json
import os
def convert_to_mcq(data):
    # Extract relevant information from the input dictionary
    question = data["question"]
    candidates = data["candidates"]
    
    # Map the candidates to letters A, B, C, ...
    options = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(candidates)])
    
    
    # Format the question and options in the desired output format
    formatted_output = f"{question}\n{options}\nAnswer with the option's letter from the given choices directly."
    
    return formatted_output
def process_qbench(image_folder="../datasets/q-bench/llvisionqa_qbench_dev/dev", json_file='../datasets/q-bench/llvisionqa_dev.json'):
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    processed_data = []
    for item in raw_data:
        image_path = os.path.join(image_folder, item["img_path"])
        if not os.path.exists(image_path):
            print(f"Image {image_path} does not exist. Skipping...")
            continue
        text = convert_to_mcq(item)
        processed_item = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text},
            ],
        }
        processed_data.append(processed_item)
    return raw_data, processed_data

if __name__ == '__main__':

    raw_data, processed_data = process_qbench()
    print(processed_data[1])