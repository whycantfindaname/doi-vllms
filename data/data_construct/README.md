# 生成训练数据meta data流程

## 1. 准备数据
所有原图片放在datasets/images/doi-images-all目录下，可以修改路径。

## 2. 生成退化图片
```
python data/data_construct/data_pipeline/visual_qwenvl.py

```
这里使用了qwenvl带的画图工具，每一个图片不同退化存在不同新的退化图上，位置在datasets/images/train_vis_dist下。对于一张图片A.jpg的low clarity，具体位置为datasets/images/train_vis_dist/A/Low_clarity.jpg，表现形式为框，每个框左上角有对应的id

## 3. 生成meta data
```
python data/data_construct/gpt_pipeline/train_gen_gpt_dist_info.py
```
保证meta数据和description数据存在后运行，每一张图的每一种退化送原图和退化图进gpt。结果存在data/meta_json/train-v1/test_dist_info_v2.json中，第一张图片已经生成完成。

# 生成benchmark

- [] 需要人工校验

## 1. 准备数据
所有原图片放在datasets/images/doi-images-all目录下，可以修改路径。

## 2. 生成退化图片
```
python data/data_construct/data_pipeline/visual_qwenvl.py

```
使用第一个函数，每一个图片不同框出一张图，位置在datasets/images/bench_vis_dist下。对于一张图片A.jpg的bbox 1，具体位置为datasets/images/bench_vis_dist/A/1.jpg，表现形式为框，左上角没有id

## 3. 生成meta data
```
python data/data_construct/gpt_pipeline/bench_gen_gpt_dist_info.py
```
保证meta数据和description数据存在后运行，每一张图的每一个框送原图和退化图进gpt。结果存在data/meta_json/benchmark-v1/test_dist_info_v2.json中，所有图已经生成完成，等待人工检验。