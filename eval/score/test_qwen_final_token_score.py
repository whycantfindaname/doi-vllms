from scorer import QwenFinalTokenScorer
import debugpy
import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception:
#     pass
model_path = '/home/liaowenjie/桌面/多模态大模型/lmms-finetune/checkpoints/qwen-vl-chat_lora-True_qlora-False-gvlmiqa-v0.1'
model_base = '/home/liaowenjie/桌面/多模态大模型/models/Qwen-VL-Chat'
model_name = 'Qwen-VL-Chat-lora'
scorer = QwenFinalTokenScorer(model_path, model_base, model_name=model_name, device='cuda')

image = '/home/liaowenjie/桌面/多模态大模型/datasets/images/doi-images-all/Portrait_v0300fg10000c4f4ck3c77u12leur4lg.png'
score = scorer([image])
print(score)