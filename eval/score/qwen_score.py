import json
import os

from scorer import QwenQAlignScorer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models/Qwen-VL-Chat', help='path to the model')
parser.add_argument('--model_base', type=str, default='../models/Qwen-VL-Chat', help='base name of the model')
parser.add_argument('--device', type=str, default='cuda:0', help='device to run the model')
parser.add_argument('--eval_file', type=str, default="eval/benchmark_2k.json", help='path to the evaluation file')
parser.add_argument('--image_folder', type=str, default='../datasets/image/gvlmiqa_bench', help='path to the folder of images')
parser.add_argument('--save_path', type=str, required=True, help='path to save the predicted answers')
args = parser.parse_args()
# gvlmiqa bench
file = args.eval_file

# 读取JSON文件
with open(file, "r", encoding="utf-8") as file:
    data = json.load(file)
img_list = []
image_dir = args.image_folder
save_path = args.save_path
os.makedirs(os.path.dirname(save_path), exist_ok=True)
for i in range(len(data)):
    try:
        image = data[i]["image"]
    except:
        image = data[i]["filename"]
    img_list.append(os.path.join(image_dir, image))

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

output = scorer(img_list)
print("Saving results to", save_path)
with open(save_path, "w") as file:
    json.dump(output, file, ensure_ascii=False, indent=4)