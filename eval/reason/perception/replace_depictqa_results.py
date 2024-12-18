import json
import os

file1 = 'results/q_bench/mplug-owl2/co-instruct.json'
file2 = 'results/q_bench/depictqa/quality_single_A_noref_qbench_qa.json'
with open(file1, 'r') as f:
    data1 = json.load(f)
with open(file2, 'r') as f:
    data2 = json.load(f)

for i in range(len(data1)):
    data1[i]['pred_ans'] = data2[i]['text'].split('\n')[0]

with open('results/q_bench/depictqa/qbench_qa.json', 'w') as f:
    json.dump(data1, f, indent=4)