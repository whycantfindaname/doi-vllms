#!/bin/bash

# 设置数据目录和模型评估的 Python 脚本路径
data_dir="../datasets/val_json"
results_dir="results/q_align"

# 循环遍历数据集
cross_datasets=("test_spaq.json"  "test_koniq.json" "test_kadid.json" "agi.json" "livec.json")

# 获取输入参数
MODEL_NAME="$1"
ORIGIN="$2"
FINETUNE="$3"

# 检查模型名称是否提供
if [ -z "$MODEL_NAME" ]; then
    echo "Please provide a model name as the first argument."
    exit 1
fi

# llava-ov
origin_llava_pred=llava_ov_score_val.json
finetune_llava_pred=llava_qalign_score.json
if [[ "$MODEL_NAME" == "llava-ov" ]]; then
    for dataset in "${cross_datasets[@]}"; do
        # 提取任务名称
        parts=(${dataset//_/ }) # 使用下划线分割
        task=${parts[1]:-${parts[0]}}  # 如果有下划线，取第二部分，否则取第一部分
        task=${task%%.*}  # 去掉扩展名

        # 定义文件路径
        save_path="$results_dir/$task/$origin_llava_pred"
        gt_json="$data_dir/$dataset"
        save_path_finetune="$results_dir/$task/$finetune_llava_pred"

        if [[ "$ORIGIN" == "true" ]]; then
            echo "Evaluating with pred_json: $save_path and gt_json: $gt_json"
            python eval/score/eval_metric.py --pred_json "$save_path" --gt_json "$gt_json"
            read -p "Press Enter to continue..."
        fi
        if [[ "$FINETUNE" == "true" ]]; then
            echo "Evaluating with pred_json: $save_path_finetune and gt_json: $gt_json"
            python eval/score/eval_metric.py --pred_json "$save_path_finetune" --gt_json "$gt_json"
            read -p "Press Enter to continue..."
        fi
    done
fi

# qwen2-vl
origin_qwen2_pred=qwen2_vl_score_val.json
finetune_qwen2_pred=qwen2_gvlmiqav0.2-onetask-train.json
if [[ "$MODEL_NAME" == "qwen2-vl" ]]; then
    for dataset in "${cross_datasets[@]}"; do
        # 提取任务名称
        parts=(${dataset//_/ }) # 使用下划线分割
        task=${parts[1]:-${parts[0]}}  # 如果有下划线，取第二部分，否则取第一部分
        task=${task%%.*}  # 去掉扩展名

        # 定义文件路径
        save_path="$results_dir/$task/$origin_qwen2_pred"
        gt_json="$data_dir/$dataset"
        save_path_finetune="$results_dir/$task/$finetune_qwen2_pred"

        if [[ "$ORIGIN" == "true" ]]; then
            echo "Evaluating with pred_json: $save_path and gt_json: $gt_json"
            python eval/score/eval_metric.py --pred_json "$save_path" --gt_json "$gt_json"
            read -p "Press Enter to continue..."
        fi
        if [[ "$FINETUNE" == "true" ]]; then
            echo "Evaluating with pred_json: $save_path_finetune and gt_json: $gt_json"
            python eval/score/eval_metric.py --pred_json "$save_path_finetune" --gt_json "$gt_json"
            read -p "Press Enter to continue..."
        fi
    done
fi

# qwen-vl
origin_qwen_pred=qwen_vl_score_val.json
finetune_qwen_pred=qwen_onetask_gvlmiqatrain_score.json
if [[ "$MODEL_NAME" == "qwen-vl" ]]; then
    for dataset in "${cross_datasets[@]}"; do
        # 提取任务名称
        parts=(${dataset//_/ }) # 使用下划线分割
        task=${parts[1]:-${parts[0]}}  # 如果有下划线，取第二部分，否则取第一部分
        task=${task%%.*}  # 去掉扩展名

        # 定义文件路径
        save_path="$results_dir/$task/$origin_qwen_pred"
        gt_json="$data_dir/$dataset"
        save_path_finetune="$results_dir/$task/$finetune_qwen_pred"

        if [[ "$ORIGIN" == "true" ]]; then
            echo "Evaluating with pred_json: $save_path and gt_json: $gt_json"
            python eval/score/eval_metric.py --pred_json "$save_path" --gt_json "$gt_json"
            read -p "Press Enter to continue..."
        fi
        if [[ "$FINETUNE" == "true" ]]; then
            echo "Evaluating with pred_json: $save_path_finetune and gt_json: $gt_json"
            python eval/score/eval_metric.py --pred_json "$save_path_finetune" --gt_json "$gt_json"
            read -p "Press Enter to continue..."
        fi
    done
fi

# internvl2
origin_intern_pred=internvl2_score_val.json
finetune_intern_pred=internvl2_lora_gvlmiqav0.2-score-train_2bs_5epoch_1e-5lr_mdp3.json
if [[ "$MODEL_NAME" == "internvl2" ]]; then
    for dataset in "${cross_datasets[@]}"; do
        # 提取任务名称
        parts=(${dataset//_/ }) # 使用下划线分割
        task=${parts[1]:-${parts[0]}}  # 如果有下划线，取第二部分，否则取第一部分
        task=${task%%.*}  # 去掉扩展名

        # 定义文件路径
        save_path="$results_dir/$task/$origin_intern_pred"
        gt_json="$data_dir/$dataset"
        save_path_finetune="$results_dir/$task/$finetune_intern_pred"

        if [[ "$ORIGIN" == "true" ]]; then
            echo "Evaluating with pred_json: $save_path and gt_json: $gt_json"
            python eval/score/eval_metric.py --pred_json "$save_path" --gt_json "$gt_json"
            read -p "Press Enter to continue..."
        fi
        if [[ "$FINETUNE" == "true" ]]; then
            echo "Evaluating with pred_json: $save_path_finetune and gt_json: $gt_json"
            python eval/score/eval_metric.py --pred_json "$save_path_finetune" --gt_json "$gt_json"
            read -p "Press Enter to continue..."
        fi
    done
fi

