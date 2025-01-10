import os
import json

bench_file = 'data/meta_json/benchmark-v1/release/dist_info_v1.json'
save_file = 'data/meta_json/benchmark-v1/test/dist_info_v1_modified.json'
os.makedirs(os.path.dirname(save_file), exist_ok=True)
with open(bench_file, 'r') as f:
    data = json.load(f)

for item in data:
    for dist in item['distortions']:
        if dist['severity'] == 'severe':
            if 'fair' in dist['region_quality_scores']:
                print(dist['region_quality_scores'])
                dist['region_quality_scores'] = "The quality is bad"
                print(dist['region_quality_scores'])
            elif 'good' in dist['region_quality_scores']:
                dist['region_quality_scores'] = "The quality is poor"
            elif 'excellent' in dist['region_quality_scores']:
                dist['region_quality_scores'] = "The quality is fair"
        
with open(save_file, 'w') as f:
    json.dump(data, f, indent=4)