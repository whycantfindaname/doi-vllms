# 介绍
eval文件夹下存放的是评估画质大模型的脚本。主要从三个角度评价一个画质大模型:
- 1.ground(画质问题定位) 
- 2.score(画质评分) 
- 3.reason(画质分析)

# 目录结构
想要顺利进行模型质量评估，除了该文件夹下的数据外，还需要下载一些必要的数据并将结果保存到如下指定目录。
```
lmms-finetune
├── eval
│   ├── README.md
│   ├── ground
│   ├── score
│   ├── reason
|—— results
│   ├── gvlmiqa_bench
│   │   ├── {model_name}
│   │   ├── grounding_results
│   ├── q_align
│   │   ├── agi
│   │   ├── kadid
│   │   ├── koniq
│   │   ├── livec
│   │   └── spaq
│   └── q_pathway
│       ├── {model_name}.json
│   ├── q_bench
│   │   ├── {mdoel_name}


datasets
|—— images
|   |—— agi-cgi
|   |—— gvlmiqa_bench
|   |—— gvlmiqa_train
|   |—— kadid10k
|   |—— spaq_koniq
|   |—— koniq
|   |—— spaq
|   |—— livec
|   |—— livefb_liveitw_aigc
|—— val_json
|   |—— q_bench_eval.json
|   |—— q_pathway_final.json
|   |—— agi.json
|   |—— test_koniq.json
|   |—— test_spaq.json
|   |—— test_kadid.json
|   |—— livec.json

q-bench
|—— llvisionqa_qbench_dev
|—— llvisionqa_dev.json
```

# 评估画质定位能力
*TODOS:*
- [ ] 支持internvl2评估

## 介绍
评估画质定位能力主要是在两种数据结构(vis和cap)下评估模型对全局、局部、全局+局部的画质问题的定位能力。目前支持[qwenvl](ground/qwen_vl.py)和[qwen2vl](ground/qwen2_vl.py)评估，评价指标为IoU(Intersection over Union)、mAP和F1-score。
<details>
<summary><b>1. vis数据结构</b></summary>
**Prompt**:  Please identify the quality issues in the image and give their bounding box coordinates both globally and locally.
**Answer**: Globally, there are no quality issues affecting the entire image.\nLocally, there are some quality issues impacting specific regions:\n\<ref>Edge aliasing effect\</ref>\<box>(762,266),(818,714)\</box>...
</details>
<details>
<summary><b>2. cap数据结构</b></summary>
**Prompt**: Describe the image content and the image quality issues with grounding.
**Answer**: This image shows a six-pin electrical connector, likely used for automotive or industrial applications, with a black rectangular casing and six round pins in a single row.\nGlobally, there are no quality issues affecting the entire image, but some issues are present in specific regions.\nLocally, there are some quality issues impacting specific regions. \<ref>Edge aliasing effect\</ref>\<box>(713,266),(771,714)\</box>... is noticeable in the right-central area, affecting the edges of the connector, and in the left-central area, impacting the casing's outline...
</details>

## 环境配置
与训练环境一致即可
## 使用
**使用流程**: 在vis_eval.sh改模型地址，结果保存地址以及相关参数后`bash vis_eval.sh`即可，最终能在results/gvlmiqa_bench/grounding_results下看到结果。
**具体流程**: 
<details>
<summary><b>1. 生成模型grounding结果</b></summary>

```python

python ./eval/ground/qwen_vl.py \
    --model_path <MODEL_PATH> \  # 替换 <MODEL_PATH> 为实际路径
    --save_path "$save_ground_path"
```

</details>

<details>
<summary><b>2. 替换gt_bbox为normalize到[0, 1000]的bbox</b></summary>

```python
python eval/ground/replace_bbox_normalize.py \
    --json_file_path "$save_ground_path"
```

</details>

<details>
<summary><b>3. 对global、local和globa+local结果分别计算IoU、mAP、F1-score并存入excel</b></summary>
</details>

# 评估画质打分能力
*TODOS:*
- [x] 支持internvl2评估
- [ ] 增加更多的IQA数据集

## 介绍
评估方式主要参考[Q-Align](https://github.com/Q-Future/Q-Align)。
支持在我们的benchmark上和五个IQA数据集(spaq,koniq,kadid,livec,agi)上评测

## 环境配置
生成打分时与训练环境一致，评测指标时需要更换环境装[pyiqa](https://github.com/chaofengc/IQA-PyTorch/tree/main)

## 使用
### 评测我们的benchmark

<details>
<summary><b>1. 生成模型打分</b></summary>

```python
python eval/score/{model_name}_score.py \
    --model_path <MODEL_PATH> \  # 替换 <MODEL_PATH> 为实际路径
    --save_path <SAVE_PATH> \  # 替换 <SAVE_PATH> 为结果保存路径
```
结果一般保存在results/gvlmiqa_bench/{model_name}下
</details>

<details>
<summary><b>2. 计算srcc和plcc等指标</b></summary>

```python
python eval/score/eval_metric.py \
    --pred_json <PRED_JSON> \  # 替换 <PRED_JSON> 为第一步生成的结果路径
```

### 评测五个IQA数据集
<details>
<summary><b>1. 生成模型打分</b></summary>

```python
python eval/score/{model_name}_score_fiveiqa.py \
    --model_path <MODEL_PATH> \  # 替换 <MODEL_PATH> 为实际路径
    --save_name <SAVE_NAME> \  # 替换 <SAVE_NAME> 为结果保存名字
```
结果一般保存在results/q_align/{iqa_dataset}下
</details>

<details>
<summary><b>2. 计算srcc和plcc等指标</b></summary>

运行[评测脚本](score/eval_fiveiqa.sh)之前需要在脚本中更改结果名字: 将finetune_{model}_pred改成第一步生成
```bash
bash eval/score/eval_fiveiqa.sh {model_name} false(baseline) true(finetune)
```
</details>


# 评估画质分析能力
*TODOS:*
- [ ] 支持internvl2评估
- [ ] 支持我们的benchmark评估

## 介绍
主要从description和perception两个角度分析模型的画质问题。
description角度主要评价模型生成的具体assessment
perception角度主要通过设计一些问答题来评价模型对图像的感知能力
具体可以参考[Q-Bench](https://github.com/Q-Future/Q-Bench)

## 环境配置
与训练环境一致

## 使用
和之前差不多

