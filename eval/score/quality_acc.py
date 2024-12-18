import argparse
import json
import os
import string
from collections import Counter

import pandas as pd
from sklearn.metrics import confusion_matrix

# 设置命令行参数
arg = argparse.ArgumentParser()
arg.add_argument("--pred_json", type=str, default="pred.json")
arg.add_argument("--gt_json", type=str, default="./eval/score/benchmark_2k.json")
arg.add_argument("--output_csv", type=str, default="None")


def assign_level(mos_score):
    if mos_score >= 4.2:
        return "excellent"
    elif mos_score >= 3.4:
        return "good"
    elif mos_score >= 2.6:
        return "fair"
    elif mos_score >= 1.8:
        return "poor"
    else:
        return "bad"


if __name__ == "__main__":
    args = arg.parse_args()

    # 加载预测和真实数据
    with open(args.pred_json, "r") as f:
        pred = json.load(f)
    with open(args.gt_json, "r") as f:
        gt = json.load(f)
    print(args.pred_json)
    print(args.gt_json)
    # 提取预测质量评分
    if isinstance(pred, list) and all(isinstance(sublist, list) for sublist in pred):
        pred = [item for sublist in pred for item in sublist]
    pred_quality = [
        max(item["logits"], key=item["logits"].get).strip() for item in pred
    ]
    print(len(pred_quality))

    # 提取真实质量评分
    gt_quality = []
    image_names = []

    # Loop through each predicted item to extract and match image names
    for pred_item in pred:
        image_name = os.path.basename(
            pred_item.get("image", pred_item.get("filename", ""))
        )
        image_names.append(image_name)

        try:
            gt_item = next(
                item
                for item in gt
                if os.path.basename(item.get("image")) == image_name 
            )
        except StopIteration:
            print(image_name)
            continue

        try:
            quality = (
                gt_item["conversations"][-1]["value"]
                .split()[-1]
                .strip(string.punctuation)
            )
        except (KeyError, IndexError, AttributeError):
            mos = gt_item.get("mos")
            quality = assign_level(mos) if mos is not None else "Unknown"

        gt_quality.append(quality)

    print(pred_quality[:10])
    print(len(gt_quality))
    print(gt_quality[:10])

    # 计算准确率
    matching_count = sum(1 for x, y in zip(pred_quality, gt_quality) if x == y)
    print("accuracy:", matching_count / len(pred_quality))

    # 定义类别
    categories = ["excellent", "good", "fair", "poor", "bad"]
    results = {category: {"TP": 0, "FP": 0, "FN": 0} for category in categories}

    # 统计每个类别的数量
    category_counts = Counter(pred_quality)
    print("model degree:", category_counts)
    category_counts1 = Counter(gt_quality)
    print("human degree:", category_counts1)

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(pred_quality, gt_quality, labels=categories)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=categories, columns=categories)

    # 显示混淆矩阵
    print("Confusion Matrix:")
    print(conf_matrix_df)

    # 计算并打印每个类别的精确率和召回率
    for p, g in zip(gt_quality, pred_quality):
        for category in categories:
            if p == category and g == category:  # True Positive
                results[category]["TP"] += 1
            elif g == category and p != category:  # False Positive
                results[category]["FP"] += 1
            elif p == category and g != category:  # False Negative
                results[category]["FN"] += 1

    for category in categories:
        TP = results[category]["TP"]
        FP = results[category]["FP"]
        FN = results[category]["FN"]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        print(f"{category}: Precision = {precision:.2f}, Recall = {recall:.2f}")

    # 优先级映射，用于计算分差
    priority = {"excellent": 5, "good": 4, "fair": 3, "poor": 2, "bad": 1}

    # 选出符合条件的图片
    selected_images = []
    for i in range(len(gt_quality)):
        diff = priority[pred_quality[i]] - priority[gt_quality[i]]
        if (pred_quality[i] in ["excellent", "good"] and diff >= 2) or (
            pred_quality[i] == "fair" and diff == 1
        ):
            selected_images.append(
                {
                    "Image Name": image_names[i],
                    "human": gt_quality[i],
                    "onealign": pred_quality[i],
                }
            )
    if args.output_csv:
    # 将选出的图片信息写入CSV文件
        selected_images_df = pd.DataFrame(selected_images)
        selected_images_df.to_csv(args.output_csv, index=False)

    print(f"Selected images written to {args.output_csv}")
