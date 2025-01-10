import os
import json
import torch
cot_path = '/home/liaowenjie/桌面/多模态大模型/lmms-finetune/results/q_align/koniq/llavaov_cot_score.json'
with open(cot_path, 'r') as f:
    cot_scores = json.load(f)

def cal_score(probabilities_dict):
    # 找到值最大的键
    max_key = max(probabilities_dict, key=probabilities_dict.get)
    # print(max_key)
    if max_key == "excellent":
        cot_score = 4.2 + 0.8 * (1 * probabilities_dict["excellent"] + 0.75 * probabilities_dict["good"] + 0.5 * probabilities_dict["fair"] + 0.25 * probabilities_dict["poor"])
    elif max_key == "good":
        cot_score = 3.4 + 0.8 * (1 * probabilities_dict["excellent"] + 0.75 * probabilities_dict["good"] + 0.5 * probabilities_dict["fair"] + 0.25 * probabilities_dict["poor"])
    elif max_key == "fair":
        cot_score = 2.6 + 0.8 * (1 * probabilities_dict["excellent"] + 0.75 * probabilities_dict["good"] + 0.5 * probabilities_dict["fair"] + 0.25 * probabilities_dict["poor"])
    elif max_key == "poor":
        cot_score = 1.8 + 0.8 * (1 * probabilities_dict["excellent"] + 0.75 * probabilities_dict["good"] + 0.5 * probabilities_dict["fair"] + 0.25 * probabilities_dict["poor"])
    else:
        cot_score = 1 + 0.8 * (1 * probabilities_dict["excellent"] + 0.75 * probabilities_dict["good"] + 0.5 * probabilities_dict["fair"] + 0.25 * probabilities_dict["poor"])
    
    return cot_score
    
for cot_score in cot_scores:
    # 提取 logits
    logits = torch.tensor(list(cot_score["logits"].values()))

    # 计算 softmax 概率
    probabilities = torch.softmax(logits, dim=0).tolist()

    # 将概率值与类别重新组合
    categories = list(cot_score["logits"].keys())
    probabilities_dict = {category.strip(): prob for category, prob in zip(categories, probabilities)}
    score = cal_score(probabilities_dict)
    cot_score['pred_mos'] = score

    # 保存结果
    with open(os.path.join(os.path.dirname(cot_path), 'test_score.json'), 'w') as f:
        json.dump(cot_scores, f, indent=4)