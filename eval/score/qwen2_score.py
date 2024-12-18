import json
import os

from scorer import Qwen2QAlignScorer
import argparse
import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception:
#     pass
# gvlmiqa bench
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models/qwen2-vl-7b-instruct', help='path to the model')
parser.add_argument('--model_base', type=str, default='../models/qwen2-vl-7b-instruct', help='base name of the model')
parser.add_argument('--device', type=str, default='cuda:0', help='device to run the model')
parser.add_argument('--eval_file', type=str, default="eval/benchmark_2k.json", help='path to the evaluation file')
parser.add_argument('--image_folder', type=str, default='../datasets/images/gvlmiqa_bench', help='path to the folder of images')
parser.add_argument('--save_path', type=str, required=True, help='path to save the predicted answers')
args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
# 读取JSON文件
with open(args.eval_file, "r", encoding="utf-8") as file:
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
model_name = 'qwen2-vl-lora' if "lora" in model_path else "qwen2-vl"
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
scorer = Qwen2QAlignScorer(model_path, model_base, model_name=model_name, device=device, level=levels)
output = []

# 每8个图像进行一次评分
for i in range(0, len(img_list), 8):
    batch = img_list[i:i + 8]  # 获取当前的8个图像
    score = scorer(batch)       # 评分
    output.append(score)        # 将结果添加到输出列表
    print("Saving results to", save_path)
    with open(save_path, "w") as file:
        json.dump(output, file, ensure_ascii=False, indent=4)