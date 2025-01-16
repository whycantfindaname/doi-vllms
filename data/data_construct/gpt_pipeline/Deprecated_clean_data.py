import json
import os 
from collections import defaultdict
from data_utils import *

if __name__ == '__main__':
    # importance_file = 'data/meta_json/train/process/train_10k_seagull_importance_normalized.json'
    # quality_file = 'data/meta_json/train/process/train_10k_seagull_quality_normalized.json'
    # data = combine_seagull_importance_quality(importance_file, quality_file)
    # with open('data/meta_json/train/process/train_10k_seagull_combined.json', 'w') as f:
    #     json.dump(data, f, indent=4, ensure_ascii=False)
    # meta_file = 'data/meta_json/train/process/train_10k_seagull_combined.json'
    # save_file = 'data/meta_json/train/process/train_10k_seagull_clean.json'
    # meta_file = 'data/meta_json/benchmark/process/benchmark_2k_normalized_seagull_cleaned_v1.json'
    save_file = 'data/meta_json/train/release/train_10k_seagull_clean.json'
    # save_file = 'data/meta_json/benchmark/release/benchmark_2k_seagull_clean.json'
    # print("Cleaning AIGC data and global bbox...")
    # with open(meta_file, 'r') as f:
    #     data = json.load(f)
    # data = clean_aigc_data(data)
    # key_count = [1]
    # while len(key_count) > 0:  # This also checks if key_count is not empty
    #     key_count, new_data = process_images(data)
    #     print(key_count)
    
    # print("Cleaning distortions...")
    # new_data = clean_dist(new_data)
    # new_data = check_sanity(new_data)
    # show_number(new_data)
    
    # with open(save_file, 'w') as f:
    #     json.dump(new_data, f, indent=4, ensure_ascii=False)
    with open(save_file, 'r') as f:
        data = json.load(f)
    # data = clean_dist(data)
    # data = check_sanity(data)
    # show_number(data)
    # with open(save_file, 'w') as f:
    #     json.dump(data, f, indent=4, ensure_ascii=False)
   
   
    # images = []
    # for item in data:
    #     dist = {**item['global'], **item['local']}
    #     if dist == {}:
    #         images.append(item['image'])
    # print(len(images))
    # dist_file = 'data/meta_json/train/release/test_dist_info_v1.json'
    # dist_data = load_json(dist_file)
    # for dist_item in dist_data:
    #     if dist_item['image'] in images:
    #         dist_data.remove(dist_item)
    # with open('data/meta_json/train/test/test_dist_info_v2.json', 'w') as f:
    #     json.dump(dist_data, f, indent=4, ensure_ascii=False)
            