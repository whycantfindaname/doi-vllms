#!/bin/bash

# 定义变量
save_ground_path="results/gvlmiqa_bench/qwenvl/qwen_gvlmiqav0.2-train-vis-ground_64bs_5epoch_3e-4lr.json"
main_header_label="qwenvl-ft-onetask-5epoch"
result_dir="results/gvlmiqa_bench/grounding_results/$(basename "$save_ground_path" .json)"  # 提取文件名作为结果目录名

# 创建输出目录
mkdir -p "$result_dir"

# # 生成 ground 结果
# python ./eval/ground/qwen_vl.py \
#     --model_path <MODEL_PATH> \  # 替换 <MODEL_PATH> 为实际路径
#     --save_path "$save_ground_path"

# 检查 qwen_vl.py 是否成功生成文件
if [ ! -f "$save_ground_path" ]; then
    echo "Error: Failed to generate ground results. File not found at $save_ground_path"
    exit 1
fi

# 替换归一化 BBox
python eval/ground/replace_bbox_normalize.py \
    --json_file_path "$save_ground_path"

# 循环处理任务类型
for task in local global all; do
    task_result_dir="$result_dir/$task"
    mkdir -p "$task_result_dir"

    # 运行 bbox_grounding_evaluate.py
    python eval/ground/bbox_grounding_evaluate.py \
        --input_filepath "$result_dir/grounding_normalized.json" \
        --type "$task" \
        --save_dir "$task_result_dir"

    # 检查评估结果是否生成
    overall_metrics_json="$task_result_dir/overall_metrics_results.json"
    if [ ! -f "$overall_metrics_json" ]; then
        echo "Error: Evaluation failed for task $task. File not found at $overall_metrics_json"
        continue
    fi

    # 运行 write2excel.py
    python eval/ground/write2excel.py \
        --input_file "$overall_metrics_json" \
        --output_file "$task_result_dir/overall_metrics_results.xlsx" \
        --main_header_label "$main_header_label"
done
