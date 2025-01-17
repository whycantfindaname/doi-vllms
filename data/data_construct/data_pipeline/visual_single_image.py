from concurrent.futures import ThreadPoolExecutor
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
import re
torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained("../models/Qwen-VL-Chat", trust_remote_code=True)

image_folder = "../datasets/images/koniq"
save_folder = "test/koniq"
image_name = '10086300493.jpg'
img_path = os.path.join(image_folder, image_name)
query = tokenizer.from_list_format([
    {'image': img_path},
])
resource = "The image depicts two individuals engaged in tree climbing or tree care activities, with one person on a branch and the other assisting, surrounded by bare branches and a few leaves against an overcast sky.\\nThere is one distortion in the image: <|object_ref_start|>Moderate low clarity<|object_ref_end|><|box_start|>(3,5),(998,995)<|box_end|> affects the tree branches, leaves, and the individuals climbing the tree. This distortion results in a general softness across the image, reducing the sharpness and detail of the tree branches and the individuals. The edges of the branches and clothing appear less defined, and the textures of the bark and leaves are less distinct. This diminishes the overall clarity and detail of the scene, making it harder to discern finer details and reducing the visual impact of the image.\\nThe <|object_ref_start|>moderate low clarity<|object_ref_end|><|box_start|>(3,5),(998,995)<|box_end|> is the key distortion affecting the image's quality. The lack of sharpness and detail in the tree branches, leaves, and individuals reduces the visual impact and clarity of the scene. The image's low-level attributes, such as contrast and texture, are also affected, leading to a less engaging visual experience. Despite this, the image maintains a coherent composition and conveys its subject matter effectively.\\nThus, the quality of the image is fair."
dist1 = "<ref>moderate low clarity</ref><box>(3,5),(998,995)</box>"
dist_list = [dist1]

for i in range(len(dist_list)):
    dist = dist_list[i]
    save_path = os.path.join(save_folder, image_name.replace('.jpg', f'_{i}.jpg'))
    history = [(query, dist)]
    image = tokenizer.draw_bbox_on_latest_picture(dist, history)
    os.makedirs(save_folder, exist_ok=True)
    if image:
        image.save(save_path)
