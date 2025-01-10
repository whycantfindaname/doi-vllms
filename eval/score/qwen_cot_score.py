import json
import os

from scorer import QwenQAlignScorer, QwenCOTScorer
# other iqadatasets
cross_datasets = ["agi.json", "test_kadid.json", "test_koniq.json", "test_spaq.json", "livec.json"]
cross_datasets = ["test_spaq.json"]
data_dir =  "../datasets/val_json"
import argparse
# gvlmiqa bench
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models/Qwen-VL-Chat', help='path to the model')
parser.add_argument('--model_base', type=str, default='../models/Qwen-VL-Chat', help='base name of the model')
parser.add_argument('--device', type=str, default='cuda:0', help='device to run the model')
parser.add_argument('--image_folder', type=str, default='../datasets/images', help='path to the folder of images')
parser.add_argument('--save_name', type=str, required=True, help='filename to save the results')
args = parser.parse_args()
save_name = args.save_name
model_path = args.model_path
model_base = args.model_base
model_name = 'qwen-vl-chat-lora' if "lora" in model_path else 'qwen-vl-chat'
levels = [
    " excellent",
    " good",
    " fair",
    " poor",
    " bad",
    " high",
    " low",
    " fine",
    " moderate",
    " decent",
    " average",
    " medium",
    " acceptable",
]
device = args.device
scorer = QwenCOTScorer(model_path, model_base, device=device, level=levels)

for dataset in cross_datasets:
    file = os.path.join(data_dir, dataset)
    with open(file, "r", encoding="utf-8") as file:
        data = json.load(file)
    img_list = []
    image_dir = args.image_folder
    parts = dataset.split(".")[0].split("_")
    if len(parts) > 1:
        task = parts[1]
    else:
        task = parts[0]
    print(task)
    save_path = f"results/q_align/{task}/{save_name}.json"
    if os.path.exists(save_path):
        save_data = json.load(open(save_path, "r"))
    else:
        save_data = []
    try:
        not_complete = [item for item in data if os.path.basename(item['image']) not in [x['filename'] for x in save_data]]
    except:
        not_complete = [item for item in data if os.path.basename(item['img_path']) not in [x['filename'] for x in save_data]]
    print(f"Number of images not complete: {len(not_complete)}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for i in range(len(not_complete)):
        try:
            image = not_complete[i]["image"]
        except:
            image = not_complete[i]['img_path']
        img_list.append(os.path.join(image_dir, image))

    output = scorer(img_list)
    print("Saving results to", save_path)
    with open(save_path, "w") as file:
        json.dump(output, file, ensure_ascii=False, indent=4)