import json
import os

from scorer import InternVL2QAlignScorer
import argparse
import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception:
#     pass
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="../models/InternVL2-8B", help='path to the model')
parser.add_argument('--model_base', type=str, default="../models/InternVL2-8B", help='base name of the model')
parser.add_argument('--eval_file', type=str, default="eval/benchmark_2k.json", help='path to the evaluation file')
parser.add_argument('--image_folder', type=str, default='../datasets/images/gvlmiqa_bench', help='path to the folder of images')
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
model_name = 'internvl2'
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
scorer = InternVL2QAlignScorer(model_path, model_base, model_name=model_name, level=levels)
output = scorer(img_list)
print("Saving results to", save_path)
with open(save_path, "w") as file:
    json.dump(output, file, ensure_ascii=False, indent=4)