from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import os

parser = argparse.ArgumentParser("merge lora weights and save model with hf format")
parser.add_argument("--save_path", type=str, default="Qwen-IQA-test")
parser.add_argument("--base_model_name_or_path", type=str, default="models/Qwen-VL-Chat")
parser.add_argument("--peft_model_path", type=str, default="checkpoints/qwen-vl-chat_lora-True_qlora-False-test")

if __name__ == "__main__":
    args = parser.parse_args()

    model_name_or_path = args.base_model_name_or_path
    save_path = args.save_path
    peft_model_path = args.peft_model_path
    os.makedirs(save_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    
    print("Applying the LoRA")
    model = model.merge_and_unload()

    print(f"Saving the target model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)



