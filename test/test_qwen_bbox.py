from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import debugpy
torch.manual_seed(1234)
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception:
#     pass

tokenizer = AutoTokenizer.from_pretrained("models/Qwen-VL-Chat", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("models/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
# print(model)

query = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPicture: <img>/home/liaowenjie/Downloads/image.webp</img>\nOutput the bounding box of objects in the image.<|im_end|>\n"
inputs = tokenizer(query, return_tensors='pt')
inputs = inputs.to(model.device)
pred = model.generate(**inputs)
response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
print(response)
# <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Generate the caption in English with grounding:<ref> Woman</ref><box>(451,379),(731,806)</box> and<ref> her dog</ref><box>(219,424),(576,896)</box> playing on the beach<|endoftext|>
# image = tokenizer.draw_bbox_on_latest_picture(query, font_size=10, linewidth=1)
# if image:
#   image.save('2.jpg')
# else:
#   print("no box")
