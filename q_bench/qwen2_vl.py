from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from prompt import process_qbench
from tqdm import tqdm
import json
raw_data, processed_data = process_qbench()

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "../models/qwen2-vl-7b-instruct", 
    torch_dtype="auto", 
    device_map="auto",
    attn_implementation="flash_attention_2"
)
print(model.hf_device_map)
print(model.dtype)  

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
min_pixels = 4*28*28
max_pixels = 3072*28*28
processor = AutoProcessor.from_pretrained('../models/qwen2-vl-7b-instruct')
# processor = AutoProcessor.from_pretrained('../models/qwen2-vl-7b-instruct', min_pixels=min_pixels, max_pixels=max_pixels)


for gt, data in tqdm(zip(raw_data,processed_data), total=len(raw_data)):
    data['content'][0]['min_pixels'] = min_pixels
    data['content'][0]['max_pixels'] = max_pixels
    messages = [data]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device).to(model.dtype)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    gt["pred_ans"] = output_text[0]
    print(gt["correct_ans"])
    print(gt["pred_ans"])

# Save the predicted answers to a file
with open('results/qwen2_vl_qbench.json', 'w') as f:
    json.dump(raw_data, f, indent=4, ensure_ascii=False)