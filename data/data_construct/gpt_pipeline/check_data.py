import json
import os 
from collections import defaultdict
from data_utils import *


if __name__ == '__main__':

    new_train_file = 'data/meta_json/train-v1/release/dist_info_v1.json'
    new_bench_file = 'data/meta_json/benchmark-v1/release/dist_info_v1.json'
    with open(new_train_file, 'r') as f:
        train_data = json.load(f)
    print(len(train_data))
    train_data = check_repeat_images(train_data)
    print(len(train_data))  
    show_processed_number(train_data)
    # with open(new_train_file, 'w') as f:
    #     json.dump(train_data, f, indent=4)
    
    with open(new_bench_file, 'r') as f:
        new_bench_data = json.load(f)
    new_bench_data = check_repeat_images(new_bench_data)
    show_processed_number(new_bench_data)
   
   