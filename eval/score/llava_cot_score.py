import json
import os

from scorer import LLaVAQAlignScorer, LLaVACOTScorer
# other iqadatasets
cross_datasets = ["agi.json", "test_spaq.json"]
# cross_datasets = ["livec.json", "test_koniq.json"]
data_dir =  "../datasets/val_json"
import argparse
# gvlmiqa bench
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='../models/llava-onevision-qwen2-7b-ov-hf', help='path to the model')
parser.add_argument('--model_base', type=str, default='../models/llava-onevision-qwen2-7b-ov-hf', help='base name of the model')
parser.add_argument('--device', type=str, default='cuda:0', help='device to run the model')
parser.add_argument('--image_folder', type=str, default='../datasets/images', help='path to the folder of images')
parser.add_argument('--save_name', type=str, required=True, help='filename to save the results')
args = parser.parse_args()

model_path = args.model_path
model_base = args.model_base
model_name = 'llava-onevision-lora' if 'lora' in model_path else 'llava-onevision'
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
scorer = LLaVACOTScorer(model_path, model_base, model_name=model_name, device=device, level=levels)

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
    # 每8个图像进行一次评分
    query = "You are an expert in image quality assessment. Your task is to assess the overall quality of the image provided.\nTo assess the image quality, you should think step by step.\n**First step**, provide a brief description of the image content in one sentence.\n**Second step**, analyze the overall image quality and visual perception.\n- If there is no distortion present in the image, focus solely on what you observe in the image and describe the image's visual aspects, such as visibility, detail discernment, clarity, brightness, lighting, composition, and texture.\n- If distortions are present, identify the distortions and briefly analyze each occurrence of every distortion type.Explain how each distortion affects the visual appearance and perception of specific objects or regions in the image.\n**Third step**, If distortions are present, identify the key distortions that have the most significant impact on the overall image quality.Provide detailed reasoning about how these key distortions affect the image's visual perception, especially regarding sharpness, clarity, and detail. Combine the analysis of key degradations and low-level attributes into a cohesive paragraph.\n**Final step**, conclude your answer with this sentence: 'Thus, the quality of the image is (one of the following five quality levels: bad, poor, fair, good, excellent)'."
    sys_prompt = "You are a helpful assistant."
    for i in range(0, len(img_list), 8):
        batch = img_list[i:i + 8]  # 获取当前的8个图像
        score = scorer(batch, sys_prompt=sys_prompt, query=query)
        output.extend(score)        # 将结果添加到输出列表
        print("Saving results to", save_path)
        with open(save_path, "w") as file:
            json.dump(output, file, ensure_ascii=False, indent=4)