import json
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('file_path', type=str, default='./dist_info_v1.json')
args = parser.parse_args()

data = json.load(open(args.file_path))
new_data = []

for i in data:
    for j in i['distortions']:
        tmp = {}
        tmp['img_path'] = i['image']
        tmp['type'] = None
        tmp['concern'] = 'severity'
        tmp['distortion'] = j['distortion']
        tmp['severity'] = j['severity']
        if j['distortion'][0].lower() in 'aeiou':
            tmp['question'] = f"There is an {j['distortion'].lower()} affecting {j['position'].lower().rstrip('.')}.How severe is it?"
        else:
            tmp['question'] = f"There is a {j['distortion'].lower()} affecting {j['position'].lower().rstrip('.')}.How severe is it?"
        tmp['candidates'] = ['Minor', 'Moderate', 'Severe', 'Not visible']
        tmp['correct_ans'] = j['severity']
        new_data.append(tmp)

with open('data/meta_json/benchmark-v1/release/severity_data.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(new_data, ensure_ascii=False, indent=4))