import argparse
import json
import os

from pyiqa.metrics.correlation_coefficient import (
    calculate_krcc,
    calculate_plcc,
    calculate_srcc,
)

arg = argparse.ArgumentParser()
arg.add_argument("--pred_json", type=str, default="pred.json")
arg.add_argument("--gt_json", type=str, default="eval/benchmark_2k.json")

if __name__ == "__main__":
    args = arg.parse_args()
    print(args.pred_json)
    print(args.gt_json)
    with open(args.pred_json, "r") as f:
        pred = json.load(f)
    with open(args.gt_json, "r") as f:
        gt = json.load(f)
    if isinstance(pred, list) and all(isinstance(sublist, list) for sublist in pred):
        pred = [item for sublist in pred for item in sublist]
    if len(pred) > len(gt):
        gt_mos = [item["mos"] for item in gt]
        pred_mos = []
        for gt_item in gt:
            pred_item = next(
                item for item in pred if item["filename"] == gt_item["image"]
            )
            pred_mos.append(pred_item["pred_mos"]["pred_mos"])
    else:
        pred_mos = [item["pred_mos"] for item in pred]

        gt_mos = []
        for pred_item in pred:
            try:
                image_name = os.path.basename(pred_item["image"])
            except:
                try:
                    image_name = os.path.basename(pred_item["img_path"])
                except:
                    image_name = os.path.basename(pred_item["filename"])
            try:
                gt_item = next(item for item in gt if os.path.basename(item["image"]) == image_name)
            except:
                gt_item = next(item for item in gt if os.path.basename(item["filename"]) == image_name)
            try:
                gt_mos.append(gt_item["mos"])
            except:
                gt_mos.append(float(gt_item["gt_score"]))
    print(len(pred_mos))
    print(pred_mos[:10])
    print(gt_mos[:10])

    srcc = calculate_srcc(gt_mos, pred_mos)
    plcc = calculate_plcc(gt_mos, pred_mos)
    # calculate_rmse(gt_mos, pred_mos)
    krcc = calculate_krcc(gt_mos, pred_mos)
    print("srcc:", srcc)
    print("plcc:", plcc)
    print("krcc:", krcc)
