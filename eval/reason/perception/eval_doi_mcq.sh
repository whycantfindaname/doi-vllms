#!/bin/bash

# 确保 Conda 已初始化
eval "$(conda shell.bash hook)"


# 激活 lmms-finetune 环境并运行脚本
conda activate lmms-finetune
python eval/reason/perception/internvl2_5.py \
    --save_path results/doi_bench/internvl2_5/internvl2_5.json \
    --eval_dataset doi-bench-mcq

# python eval/reason/perception/internvl2.py \
#     --save_path results/doi_bench/internvl2/internvl2_position.json \
#     --eval_dataset doi-bench-saq

python eval/reason/perception/qwen_vl.py \
    --save_path results/doi_bench/qwenvl/qwenvl.json \
    --eval_dataset doi-bench-mcq

# python eval/reason/perception/qwen2_vl.py \
#     --save_path results/doi_bench/qwen2vl/qwen2vl_position.json \
#     --eval_dataset doi-bench-saq

python eval/reason/perception/llava_ov.py \
    --save_path results/doi_bench/llava/llavaov.json \
    --eval_dataset doi-bench-mcq

# 激活 qalign 环境并运行脚本
conda activate qalign
python eval/reason/perception/co-instruct.py
conda deactivate