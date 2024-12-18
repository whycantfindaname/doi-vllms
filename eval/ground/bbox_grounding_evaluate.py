import json
import re
from collections import defaultdict
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input_filepath', type=str, required=True)
parser.add_argument('--iou_threshold', type=float, default=0.5)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--type', type=str, choices=['global', 'local', 'all'])


# Load JSON data
def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_pred_bboxes(pred_str):
    patterns = [
        r"<ref>(.*?)</ref>(.*?)((?=<ref>|$))",  # 捕获标签及其所有关联的框
        r"<\|object_ref_start\|>(.*?)<\|object_ref_end\|><\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>",
    ]
    
    bboxes = defaultdict(list)
    
    # 遍历每种模式
    for pattern in patterns:
        matches = re.findall(pattern, pred_str, re.DOTALL)
        if matches:
            # 第一种匹配模式：处理 <ref> 标签及多框的情况
            if len(matches[0]) == 3:
                for label, box_content, _ in matches:
                    label = label.strip().title()
                    box_pattern = r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>"
                    box_matches = re.findall(box_pattern, box_content)
                    for box in box_matches:
                        tl_x, tl_y, br_x, br_y = map(int, box)
                        bboxes[label].append({"tl": {"x": tl_x, "y": tl_y}, "br": {"x": br_x, "y": br_y}})
            
            # 第二种匹配模式：每个框都包含完整的标签与坐标
            else:
                for match in matches:
                    label = match[0].strip().title()
                    tl_x, tl_y, br_x, br_y = map(int, match[1:])
                    bboxes[label].append({"tl": {"x": tl_x, "y": tl_y}, "br": {"x": br_x, "y": br_y}})
            
            break  # 如果当前模式成功匹配，则无需继续匹配后续模式
    
    return bboxes

# 提取全局框的函数
def extract_global_bboxes(pred_str):
    pattern = r"Globally.*?<ref>(.*?)</ref>(.*?)Locally"  # 匹配全局范围和框内容
    bboxes = defaultdict(list)

    match = re.search(pattern, pred_str, re.DOTALL)  # 匹配全局部分
    if match:
        label = match.group(1).strip().title()  # 提取退化类别标签
        box_content = match.group(2)
        boxes = re.findall(r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>", box_content)
        for box in boxes:
            tl_x, tl_y, br_x, br_y = map(int, box)
            bboxes[label].append({"tl": {"x": tl_x, "y": tl_y}, "br": {"x": br_x, "y": br_y}})
    
    return bboxes

# 提取局部框的函数
def extract_local_bboxes(pred_str):
    pattern = r"Locally.*?<ref>(.*?)</ref>(.*?)$"  # 匹配局部范围和框内容
    bboxes = defaultdict(list)

    matches = re.findall(r"<ref>(.*?)</ref>(.*?)(?=<ref>|$)", pred_str, re.DOTALL)  # 匹配每个退化类别
    for label, box_content in matches:
        label = label.strip().title()  # 提取退化类别标签
        boxes = re.findall(r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>", box_content)
        for box in boxes:
            tl_x, tl_y, br_x, br_y = map(int, box)
            bboxes[label].append({"tl": {"x": tl_x, "y": tl_y}, "br": {"x": br_x, "y": br_y}})
    
    return bboxes

# Compute IoU (Intersection over Union) for two bounding boxes
def compute_iou(boxA, boxB):
    xA = max(boxA['tl']['x'], boxB['tl']['x'])
    yA = max(boxA['tl']['y'], boxB['tl']['y'])
    xB = min(boxA['br']['x'], boxB['br']['x'])
    yB = min(boxA['br']['y'], boxB['br']['y'])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA['br']['x'] - boxA['tl']['x'] + 1) * (boxA['br']['y'] - boxA['tl']['y'] + 1)
    boxBArea = (boxB['br']['x'] - boxB['tl']['x'] + 1) * (boxB['br']['y'] - boxB['tl']['y'] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# # Calculate precision, recall, and F1 score for a single label based on IoU threshold
# def calculate_label_metrics(gt_boxes, pred_boxes, iou_threshold, class_weight=1.0):
#     true_positive = 0
#     total_gt = len(gt_boxes)
#     total_pred = len(pred_boxes)
    
#     for pred in pred_boxes:
#         ious = [compute_iou(pred, gt) for gt in gt_boxes]
#         max_iou = max(ious) if ious else 0
#         if max_iou >= iou_threshold:
#             true_positive += 1

#     false_negative = total_gt - true_positive
#     false_positive = total_pred - true_positive
    
#     # Precision calculation
#     precision = true_positive / (true_positive + false_positive + 1e-6)
    
#     # Recall calculation
#     recall = true_positive / (total_gt + 1e-6)
    
#     # F1 Score calculation
#     f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
#     # Apply class weight
#     weighted_f1_score = f1_score * class_weight
    
#     return precision, recall, weighted_f1_score

def calculate_label_metrics(gt_boxes, pred_boxes, iou_threshold, class_weight=1.0):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    total_gt = len(gt_boxes)
    total_pred = len(pred_boxes)
    
    # To track which ground truth boxes have been matched
    matched_gt_boxes = set()

    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i not in matched_gt_boxes:
                iou = compute_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
        
        # If best IoU is above the threshold, consider it a true positive
        if best_iou >= iou_threshold and best_gt_idx != -1:
            true_positive += 1
            matched_gt_boxes.add(best_gt_idx)  # Mark this ground truth box as matched
        else:
            false_positive += 1

    # False negatives are the ground truth boxes that were not matched
    false_negative = total_gt - len(matched_gt_boxes)

    # Precision calculation
    precision = true_positive / (true_positive + false_positive + 1e-6)
    
    # Recall calculation
    recall = true_positive / (true_positive + false_negative + 1e-6)  
    
    # F1 Score calculation
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # Apply class weight
    weighted_f1_score = f1_score * class_weight
    
    return precision, recall, weighted_f1_score



def calculate_map(gt_boxes, pred_boxes, iou_thresholds=[0.5]):
    ap_list = []

    # Iterate over each IoU threshold to calculate AP
    for iou_threshold in iou_thresholds:
        true_positive = 0
        false_positive = 0
        total_gt = len(gt_boxes)
        total_pred = len(pred_boxes)
        
        # To track which ground truth boxes have been matched
        matched_gt_boxes = set()

        # Iterate through all predicted boxes
        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for i, gt in enumerate(gt_boxes):
                if i not in matched_gt_boxes:  # Avoid matching the same GT box multiple times
                    iou = compute_iou(pred, gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = i

            # If best IoU is above the threshold, consider it a true positive
            if best_iou >= iou_threshold and best_gt_idx != -1:
                true_positive += 1
                matched_gt_boxes.add(best_gt_idx)  # Mark this ground truth box as matched
            else:
                false_positive += 1

        # Calculate false negatives (GTs that were not matched)
        false_negative = total_gt - len(matched_gt_boxes)

        # Calculate precision and recall
        precision = true_positive / (true_positive + false_positive + 1e-6)
        recall = true_positive / (total_gt + 1e-6)

        # Append AP for current IoU threshold
        ap_list.append(precision * recall)

    # Calculate mean AP across all IoU thresholds
    mean_ap = np.mean(ap_list) if ap_list else 0.0
    return mean_ap


# Calculate IoU for each image and provide an overall score
def calculate_image_iou(pred_vis_bboxes, pred_cap_bboxes):
    iou_results = {}
    total_iou = 0
    count = 0
    
    for label in pred_vis_bboxes.keys() | pred_cap_bboxes.keys():
        vis_boxes = pred_vis_bboxes.get(label, [])
        cap_boxes = pred_cap_bboxes.get(label, [])
        
        if vis_boxes and cap_boxes:
            ious = [compute_iou(vis_box, cap_box) for vis_box in vis_boxes for cap_box in cap_boxes]
            avg_iou = np.mean(ious) if ious else 0.0
            iou_results[label] = avg_iou
            total_iou += avg_iou
            count += 1
        elif vis_boxes:
            iou_results[label] = {
                "iou": 0.0,
                "missed_detection": "Only pred_vis detected this label"
            }
        elif cap_boxes:
            iou_results[label] = {
                "iou": 0.0,
                "missed_detection": "Only pred_cap detected this label"
            }

    overall_iou = total_iou / count if count > 0 else 0.0
    return iou_results, overall_iou

# Main function to calculate accuracy and mAP for all images and labels
def calculate_metrics(data, iou_threshold, type):
    metrics_scores = {}
    label_metrics_scores = defaultdict(lambda: defaultdict(list))
    label_counts = defaultdict(lambda: defaultdict(int))
    label_map_scores = defaultdict(lambda: defaultdict(list))
    overall_image_ious = [] 
    
    for entry in data:
        image_id = entry['image']
        global_bboxes = {k.title(): v for k, v in entry.get('global', {}).items()}  # Standardize to title case
        local_bboxes = {k.title(): v for k, v in entry.get('local', {}).items()}  # Standardize to title case
        pred_vis_bboxes = extract_pred_bboxes(entry.get('pred_vis', ""))
        pred_cap_bboxes = extract_pred_bboxes(entry.get('pred_cap', ""))

        pred_vis_global_bboxes = extract_global_bboxes(entry.get('pred_vis', ""))
        pred_vis_local_bboxes = extract_local_bboxes(entry.get('pred_vis', ""))

        pred_cap_global_bboxes = extract_global_bboxes(entry.get('pred_cap', ""))
        pred_cap_local_bboxes = extract_local_bboxes(entry.get('pred_cap', ""))
        
        # TODO: Add support for global OR local bboxes 
        # Calculate metrics for pred_vis against global and local
        image_label_metrics_vis = {}
        for label in global_bboxes.keys() | local_bboxes.keys() | pred_vis_bboxes.keys():
            if type == 'all':
                gt_boxes = global_bboxes.get(label, []) + local_bboxes.get(label, [])
                pred_boxes = pred_vis_bboxes.get(label, [])
            elif type == 'global':
                gt_boxes = global_bboxes.get(label, [])
                pred_boxes = pred_vis_global_bboxes.get(label, [])
            elif type == 'local':
                gt_boxes = local_bboxes.get(label, [])
                pred_boxes = pred_vis_local_bboxes.get(label, [])

            precision, recall, f1_score = calculate_label_metrics(gt_boxes, pred_boxes, iou_threshold)
            map_score = calculate_map(gt_boxes, pred_boxes)
            image_label_metrics_vis[label] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "mAP": map_score
            }
            label_metrics_scores['pred_vis'][label].append(f1_score)
            label_map_scores['pred_vis'][label].append(map_score)
            label_counts['pred_vis'][label] += len(pred_boxes)
        
        # Calculate metrics for pred_cap against global and local
        image_label_metrics_cap = {}
        for label in global_bboxes.keys() | local_bboxes.keys() | pred_cap_bboxes.keys():
            if type == 'all':
                gt_boxes = global_bboxes.get(label, []) + local_bboxes.get(label, [])
                pred_boxes = pred_cap_bboxes.get(label, [])
            elif type == 'global':
                gt_boxes = global_bboxes.get(label, [])
                pred_boxes = pred_cap_global_bboxes.get(label, [])
            elif type == 'local':
                gt_boxes = local_bboxes.get(label, [])
                pred_boxes = pred_cap_local_bboxes.get(label, [])
            precision, recall, f1_score = calculate_label_metrics(gt_boxes, pred_boxes, iou_threshold)
            map_score = calculate_map(gt_boxes, pred_boxes)
            image_label_metrics_cap[label] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "mAP": map_score
            }
            label_metrics_scores['pred_cap'][label].append(f1_score)
            label_map_scores['pred_cap'][label].append(map_score)
            label_counts['pred_cap'][label] += len(pred_boxes)
        
        # Calculate IoU for pred_vis and pred_cap
        iou_results, overall_iou = calculate_image_iou(pred_vis_bboxes, pred_cap_bboxes)
        overall_image_ious.append(overall_iou)
        
        metrics_scores[image_id] = {
            "pred_vis": image_label_metrics_vis,
            "pred_cap": image_label_metrics_cap,
            "iou_results": iou_results,
            "overall_iou": overall_iou
        }
    
    # Calculate overall metrics for each label and type (pred_vis and pred_cap)
    label_f1_scores = {
        pred_type: {label: np.mean(scores) for label, scores in labels.items()}
        for pred_type, labels in label_metrics_scores.items()
    }
    label_map = {
        pred_type: {label: np.mean(scores) if scores else 0.0 for label, scores in labels.items()}
        for pred_type, labels in label_map_scores.items()
    }
    overall_iou_score = np.mean(overall_image_ious) if overall_image_ious else 0.0

    return metrics_scores, label_f1_scores, label_map, label_metrics_scores, label_counts, overall_iou_score

# Calculate overall IoU, mAP, and F1 for each degradation type
def calculate_overall_metrics(label_metrics_scores, label_map_scores, overall_iou_score):
    overall_metrics = {}
    
    for pred_type, labels in label_metrics_scores.items():
        overall_metrics[pred_type] = {
            "IoU": {},
            "mAP": np.mean([np.mean(scores) for scores in label_map_scores[pred_type].values() if scores]),
            "AP": {label: (np.mean(scores) if scores else 0.0) for label, scores in label_map_scores[pred_type].items()},
            "average_F1": np.mean([np.mean(scores) for scores in labels.values() if scores]),
            "F1": {label: np.mean(scores) for label, scores in labels.items() if scores}
        }
        for label, scores in labels.items():
            ious = [score for score in scores if score > 0]
            overall_metrics[pred_type]["IoU"][label] = np.mean(ious) if ious else 0.0
    
    overall_metrics['overall_iou_between_models'] = overall_iou_score
    
    return overall_metrics

# Save metrics results to a JSON file
def save_metrics_results(metrics_scores, label_f1_scores, label_map, output_filepath):
    with open(output_filepath, 'w') as f:
        json.dump({
            "image_scores": metrics_scores,
            "label_f1_scores": label_f1_scores,
            "label_map": label_map
        }, f, indent=4, ensure_ascii=False)

# Save overall IoU, mAP, and F1 results to a JSON file
def save_overall_metrics(overall_metrics, output_filepath):
    with open(output_filepath, 'w') as f:
        json.dump(overall_metrics, f, indent=4, ensure_ascii=False)

# Save extracted bounding boxes to a JSON file
def save_all_bboxes(data, output_filepath, type):
    extracted_bboxes = {}
    
    for entry in data:
        image_id = entry['image']
        global_bboxes = {k.title(): v for k, v in entry.get('global', {}).items()}  # Standardize to title case
        local_bboxes = {k.title(): v for k, v in entry.get('local', {}).items()}  # Standardize to title case


        if type == 'all':
            pred_vis_bboxes = extract_pred_bboxes(entry.get('pred_vis', ""))
            pred_cap_bboxes = extract_pred_bboxes(entry.get('pred_cap', ""))
        elif type == 'global':
            pred_vis_bboxes = extract_global_bboxes(entry.get('pred_vis', ""))
            pred_cap_bboxes = extract_global_bboxes(entry.get('pred_cap', ""))
        elif type == 'local':
            pred_vis_bboxes = extract_local_bboxes(entry.get('pred_vis', ""))
            pred_cap_bboxes = extract_local_bboxes(entry.get('pred_cap', ""))
        
        extracted_bboxes[image_id] = {
            "global": global_bboxes,
            "local": local_bboxes,
            "pred_vis": dict(pred_vis_bboxes),
            "pred_cap": dict(pred_cap_bboxes)
        }
    
    with open(output_filepath, 'w') as f:
        json.dump(extracted_bboxes, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    import os
    args = parser.parse_args()
    input_filepath = args.input_filepath
    iou_threshold = args.iou_threshold
    save_dir = args.save_dir
    # Load the data
    if save_dir is not None:
        metrics_output_filepath = f"{save_dir}/metrics_results.json"
        bbox_output_filepath = f"{save_dir}/all_bboxes.json"
        overall_metrics_output_filepath = f"{save_dir}/overall_metrics_results.json"
    else:
        metrics_output_filepath = "./metrics_results.json"
        bbox_output_filepath = "./all_bboxes.json"
        overall_metrics_output_filepath = "./overall_metrics_results.json"

    os.makedirs(os.path.dirname(metrics_output_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(bbox_output_filepath), exist_ok=True) 
    os.makedirs(os.path.dirname(overall_metrics_output_filepath), exist_ok=True)

    data = load_json(input_filepath)

    # Calculate metrics for each image and each label
    metrics_scores, label_f1_scores, label_map, label_metrics_scores, label_counts, overall_iou_score = calculate_metrics(data, iou_threshold, type=args.type)
    
    # Calculate overall metrics (IoU, mAP, and F1)
    overall_metrics = calculate_overall_metrics(label_metrics_scores, label_map, overall_iou_score)
    
    # Save the results
    save_metrics_results(metrics_scores, label_f1_scores, label_map, metrics_output_filepath)
    print(f"Metrics results saved to {metrics_output_filepath}")
    
    # Save overall IoU, mAP, and F1 results
    save_overall_metrics(overall_metrics, overall_metrics_output_filepath)
    print(f"Overall metrics results saved to {overall_metrics_output_filepath}")
    
    # Save all bounding boxes in one file
    save_all_bboxes(data, bbox_output_filepath,type=args.type)
    print(f"All bounding boxes saved to {bbox_output_filepath}")
    
    # Print label counts and mAP for each prediction type
    for pred_type, labels in label_counts.items():
        print(f"Label counts for {pred_type}:")
        for label, count in labels.items():
            print(f"  {label}: {count}")
        print(f"mAP for {pred_type}: {overall_metrics[pred_type]['mAP']}")