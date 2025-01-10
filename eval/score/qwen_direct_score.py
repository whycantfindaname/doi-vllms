import json
import os

from scorer import QwenQAlignScorer, QwenFinalTokenScorer
# other iqadatasets
cross_datasets = ["agi.json", "test_kadid.json", "test_koniq.json", "livec.json"]
# cross_datasets = ["test_spaq.json"]
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
scorer = QwenQAlignScorer(model_path, model_base, device=device, level=levels)

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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for i in range(len(data)):
        try:
            image = data[i]["image"]
        except:
            image = data[i]['img_path']
        img_list.append(os.path.join(image_dir, image))

    output = scorer(img_list)
    print("Saving results to", save_path)
    with open(save_path, "w") as file:
        json.dump(output, file, ensure_ascii=False, indent=4)