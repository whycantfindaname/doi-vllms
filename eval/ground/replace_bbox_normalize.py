import json
import os
from PIL import Image
from collections import OrderedDict

from PIL import Image, ExifTags

# meta_data
# 获取图片尺寸函数，严格读取实际宽高
def get_image_size(image_folder, filename):
    image_path = os.path.join(image_folder, filename)
    with Image.open(image_path) as img:
        width, height = img.size  # 初始获取宽和高

        # 获取EXIF方向信息，如果有
        try:
            exif = img._getexif()
            orientation_key = [key for key, val in ExifTags.TAGS.items() if val == 'Orientation'][0]
            orientation = exif.get(orientation_key, 1)

            # 如果图像需要旋转以正确显示，调整宽高
            if orientation in [6, 8]:  # 6和8表示图像被旋转了90度或270度
                width, height = height, width

        except (AttributeError, KeyError, IndexError, TypeError):
            # 没有EXIF信息的情况，忽略
            pass
        return width, height  # 返回实际宽和高


# 归一化坐标函数，并将超出1000的坐标写入日志文件
def normalize_coordinates(box, image_width, image_height, filename, log_file):
    x1, y1, x2, y2 = box
    normalized = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000)
    ]
    
    # 仅记录超过 1000 的坐标到日志
    if any(coord > 1000 for coord in normalized):
        log_file.write(f"Exceeded 1000 - Image: {image}, Original box: {box}, "
                       f"Image size: ({image_width}, {image_height}), Normalized box: {normalized}\n")
    
    return normalized

def replace_bbox(json_file_path, meta_file_path):
    # 加载 JSON 数据
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 加载元数据
    with open(meta_file_path, 'r', encoding='utf-8') as file:
        meta_data = json.load(file)

    # 创建一个列表存储处理后的数据
    new_data = []

    for item in data:
        for item1 in meta_data:
            if item1["filename"] == item["image"]:
                item["global"] = item1["global"]
                item["local"] = item1["local"]
                break
        new_data.append(item)
    # 将更新后的数据保存到新的 JSON 文件
    with open(json_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(new_data, outfile, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file_path', type=str, default='./results/gvlmiqa_bench/qwenvl/qwen_gvlmiqav0.2-train-ground_64bs_5epoch_3e-4lr.json', help='path to the original JSON file')
    parser.add_argument('--meta_file_path', type=str, default='./eval/ground/meta_data_v1.json', help='path to the meta data file')
    parser.add_argument('--image_folder', type=str, default='../datasets/images/gvlmiqa_bench/', help='path to the image folder')
    args = parser.parse_args()


    # 设置路径
    json_file_path = args.json_file_path  # 原始 JSON 文件路径
    meta_file_path = args.meta_file_path  # 元数据文件路径
    image_folder = args.image_folder  # 图像文件夹路径
    output_file_dir = os.path.basename(json_file_path).split('.json')[0]  # 输出文件夹路径
    output_file_path = f'results/gvlmiqa_bench/grounding_results/{output_file_dir}/grounding_normalized.json'  # 输出 JSON 文件路径
    log_file_path = f'results/gvlmiqa_bench/grounding_results/{output_file_dir}/normalization_log.txt'  # 日志文件路径
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # 创建输出文件夹

    # replace_bbox(json_file_path, meta_file_path)
    # 加载 JSON 数据
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    # 打开日志文件
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        # 创建一个列表存储处理后的数据
        new_data = []
        
        for item in data:
            image = item["image"]

            try:
                # 获取图像尺寸
                width, height = get_image_size(image_folder, image)
                
                # 标记是否包含超出 1000 的坐标
                has_exceeding_coordinates = False
                
                # 创建一个有序字典并添加图像字段
                ordered_item = OrderedDict()
                ordered_item["image"] = image

                # 检查 global 和 local 块的所有坐标
                for region_type in ["global", "local"]:
                    for effect in item.get(region_type, {}):
                        for region in item[region_type][effect]:
                            # 获取原始坐标
                            x1 = region["tl"]["x"]
                            y1 = region["tl"]["y"]
                            x2 = region["br"]["x"]
                            y2 = region["br"]["y"]

                            # 归一化坐标，并仅在超出1000时写入日志
                            normalized_coords = normalize_coordinates([x1, y1, x2, y2], width, height, image, log_file)
                            
                            # 检查是否存在超过 1000 的坐标
                            if any(coord > 1000 for coord in normalized_coords):
                                has_exceeding_coordinates = True

                            # 更新 JSON 数据中的坐标
                            region["tl"]["x"], region["tl"]["y"] = normalized_coords[0], normalized_coords[1]
                            region["br"]["x"], region["br"]["y"] = normalized_coords[2], normalized_coords[3]

                # 如果当前项有坐标超过 1000，添加图像尺寸信息
                if has_exceeding_coordinates:
                    ordered_item["image_width"] = width
                    ordered_item["image_height"] = height

                # 复制其他字段
                for key in item:
                    if key != "filename":
                        ordered_item[key] = item[key]

                # 将处理后的数据项添加到列表
                new_data.append(ordered_item)

            except FileNotFoundError:
                log_file.write(f"Image {image} not found in {image_folder}\n")

    # 将更新后的数据保存到新的 JSON 文件
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(new_data, outfile, ensure_ascii=False, indent=4)
