import json

gpt_dist_path = 'dataset/meta_json/benchmark/benchmark_2k_cleaned_dist_info.json'
seagull_path = 'dataset/meta_json/benchmark/benchmark_2k_normalized_seagull_cleaned.json'
save_path = 'dataset/meta_json/benchmark/benchmark_2k_normalized_new.json'
with open(gpt_dist_path, 'r') as f:
    gpt_dist_data = json.load(f)

with open(seagull_path, 'r') as f:
    seagull_data = json.load(f)

new_data = []

for seagull_item in seagull_data:
    try:
        gpt_dist_item = next(item for item in gpt_dist_data if item['filename'] == seagull_item['image'])
    except StopIteration:
        # print(f"{seagull_item['image']} not found in gpt_dist_data")
        continue
    image = gpt_dist_item['filename']
    meta_bboxes = gpt_dist_item['bboxes'].copy()

    mos = seagull_item['mos']
    level = seagull_item['level']
    width = seagull_item['width']
    height = seagull_item['height']
    global_dist = seagull_item['global']
    local_dist = seagull_item['local']
    dist_class = {**global_dist, **local_dist}
    bbox_count = 0
    try:
        for distortion, bboxes in dist_class.items():
            for bbox in bboxes:
                coordinates = [
                    bbox["tl"]["x"],
                    bbox["tl"]["y"],
                    bbox["br"]["x"],
                    bbox["br"]["y"]
                ]
                try:
                    region_quality_score = bbox["seagull"]["quality score"]
                    region_importance_score = bbox["seagull"]["importance score"]
                except:
                    region_quality_score = None
                    region_importance_score = None
                meta_bboxes[bbox_count]['distortion'] = distortion
                meta_bboxes[bbox_count]['coordinates'] = coordinates
                meta_bboxes[bbox_count]['region_quality_scores'] = region_quality_score
                meta_bboxes[bbox_count]['region_importance_score'] = region_importance_score
                bbox_count += 1
    except Exception as e:
        print(f"Error processing {seagull_item['image']}: {e}")
        gpt_dist_data.remove(gpt_dist_item)
        continue
    new_data.append({
        'image': image,
        'mos': mos,
        'level': level,
        'width': width,
        'height': height,
        'bboxes': meta_bboxes
    })

with open(save_path, 'w') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)
with open(gpt_dist_path, 'w') as f:
    json.dump(gpt_dist_data, f, indent=4, ensure_ascii=False)

 
            
