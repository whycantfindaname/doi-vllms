from scorer import QwenCOTScorer, Qwen2COTScorer, LLaVACOTScorer, InternVLCOTScorer
import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception:
#     pass
# model_path = '/home/liaowenjie/桌面/多模态大模型/models/Qwen-VL-Chat'
# model_base = '/home/liaowenjie/桌面/多模态大模型/models/Qwen-VL-Chat'
# model_name = 'Qwen-VL-Chat'
# scorer = QwenCOTScorer(model_path, model_base, model_name=model_name, device='cuda')

model_path = '/home/liaowenjie/桌面/多模态大模型/models/qwen2-vl-7b-instruct'
model_base = '/home/liaowenjie/桌面/多模态大模型/models/qwen2-vl-7b-instruct'
model_name = 'qwen2-vl-7b-instruct'
scorer = Qwen2COTScorer(model_path, model_base, model_name=model_name, device='cuda')

# model_path = '/home/liaowenjie/桌面/多模态大模型/models/llava-onevision-qwen2-7b-ov-hf'
# model_base = '/home/liaowenjie/桌面/多模态大模型/models/llava-onevision-qwen2-7b-ov-hf'
# model_name = 'llava-onevision-qwen2-7b-ov-hf'
# scorer = LLaVACOTScorer(model_path, model_base, model_name=model_name, device='cuda')

# model_path = model_base = '../models/InternVL2_5-8B'
# model_name = 'InternVL2_5-8B'
# scorer = InternVLCOTScorer(model_path, model_base, model_name=model_name, device='cuda')

image = '/home/liaowenjie/桌面/多模态大模型/datasets/images/doi-images-all/Portrait_v0300fg10000c4f4ck3c77u12leur4lg.png'
sys = 'You are a helpful assistant.'
query = "You are an expert in image quality assessment. Your task is to assess the overall quality of the image provided.\nTo assess the image quality, you should think step by step.\n**First step**, provide a brief description of the image content.\n**Second step**, analyze the overall image quality and visual perception.\n- If there is no distortion present in the image, focus solely on what you observe in the image and describe the image's visual aspects, such as visibility, detail discernment, clarity, brightness, lighting, composition, and texture.\n- If distortions are present, identify the distortions and briefly analyze each occurrence of every distortion type.Explain how each distortion affects the visual appearance and perception of specific objects or regions in the image.\n**Third step**, If distortions are present, identify the key distortions that have the most significant impact on the overall image quality.Provide detailed reasoning about how these key distortions affect the image's visual perception, especially regarding sharpness, clarity, and detail. Combine the analysis of key degradations and low-level attributes into a cohesive paragraph.\n**Final step**, conclude your answer with this sentence: 'Thus, the quality of the image is (one of the following five quality levels: bad, poor, fair, good, excellent)'."
question = "Assess the overall quality of the image provided in detail."
score = scorer([image], sys_prompt=sys, query=query)
# score = scorer([image])