import json
import os

from scorer import InternVLQAlignScorer
import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception:
#     pass

# other iqadatasets
cross_datasets = ["agi.json", "test_koniq.json", "test_spaq.json", "livec.json"]
data_dir =  "../datasets/val_json"
import argparse
# gvlmiqa bench
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="../models/InternVL2-8B", help='path to the model')
parser.add_argument('--model_base', type=str, default="../models/InternVL2-8B", help='base name of the model')
parser.add_argument('--image_folder', type=str, default='../datasets/images', help='path to the folder of images')
parser.add_argument('--save_name', type=str, required=True, help='filename to save the results')
args = parser.parse_args()

model_path = args.model_path
model_base = args.model_base
model_name = os.path.basename(model_path)
levels = [
    " excellent",
    " good",
    " fair",
    " poor",
    " bad",
    " high",
    " fine",
    " moderate",
    " decent",
    " average",
    " medium",
    " acceptable",
]
scorer = InternVLQAlignScorer(model_path, model_base, model_name=model_name, level=levels)

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
    save_path = f"results/q_align/{task}/{args.save_name}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as file:
            output = json.load(file)
    else:           
        output = []
        fail_img = []
    not_complete = data[len(output):]

    for i in range(len(not_complete)):
        try:
            image = not_complete[i]["image"]
        except:
            image = not_complete[i]['img_path']
        img_list.append(os.path.join(image_dir, image))
    query = "You are an expert in image quality assessment. Please rate the quality of the image in a single sentence: 'The quality of the image is (one of the following five quality levels: bad, poor, fair, good, excellent)'"
    # 每8个图像进行一次评分
    for i in range(0, len(img_list), 8):
        batch = img_list[i:i + 8]  # 获取当前的8个图像
        score = scorer(batch, query=query)
        output.extend(score)        # 将结果添加到输出列表
        print("Saving results to", save_path)
        with open(save_path, "w") as file:
            json.dump(output, file, ensure_ascii=False, indent=4)