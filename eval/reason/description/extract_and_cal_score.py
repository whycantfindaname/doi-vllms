import os
import json
from typing import List
def extract_score(file_path: str) -> List[int]:
    scores = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        for item in data[0:999]:
            # Check if 'scores' has exactly 5 elements
            # if len(item['scores']) != 5:
            #     # print(f"Warning: Expected 5 scores but got {len(item['scores'])} for {file_path}")
            #     # print(item)
            #     continue  # Skip this item and continue processing other items
            scores.extend(item['scores'])
    return scores

def cal_score(scores: List[int]) -> float:
    # Check if the scores list contains only integers
    if not all(isinstance(score, int) for score in scores):
        raise ValueError("All elements in the scores list must be integers.")
    print(len(scores)) 
    total_score = sum(scores)  # Sum all scores at once
    return total_score / len(scores)

def save_final_data(file_path: str):
    with open(file_path, 'r') as f:
        data = json.load(f)
    data = data[0:999]  # Only keep the first 1000 items
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='results/q_pathway/completeness')
    args = parser.parse_args()
    data_dir = args.data_dir
    file_list = os.listdir(data_dir)
    for file in file_list:
        if file.endswith('.json') == False:
            continue
        file_name = file.split('.json')[0]
        file_path = os.path.join(data_dir, file)
        save_final_data(file_path)
        scores = extract_score(file_path)
        if len(scores) == 0:
            continue
        print(f'{file_name}: {cal_score(scores)}')
        
            