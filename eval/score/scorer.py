import os
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
from builder import (
    internvl2_load_image,
    load_image,
    load_pretrained_model,
)
from tqdm import tqdm

llava_template = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + ' '}}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>' }}"
    "{% endfor %}"
    "{# Render all video then #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'video') %}"
    "{{ '<video>' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% if message['role'] != 'assistant' %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ '\n' + content['text'] }}"
    "{% endfor %}"
    "{% else %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{% generation %}"
    "{{ '\n' + content['text'] }}"
    "{% endgeneration %}"
    "{% endfor %}"
    "{% endif %}"
    "{% if message['role'] != 'assistant' and not loop.last %}"
    "{{'<|im_end|>'}}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)


class LLaVAQAlignScorer(nn.Module):
    def __init__(
        self,
        model_path,
        model_base,
        device,
        level=[" excellent", " good", " fair", " poor", " bad"],
        model_name=None,
    ):
        """
        Initializes the LLaVAQAlignScorer class.

        Args:
            model_path (str): The path to the pretrained model.
            model_base (str): The base model to be used.
            device (torch.device): The device on which to load the model (e.g., 'cpu' or 'cuda' or 'cuda:0'), if device is "cuda", device_map will be "auto", otherwise, device_map will be device.
            level (List[str]): A list of strings representing the quality levels for scoring. Defaults to ["excellent", "good", "fair", "poor", "bad"].
            model_name (str, optional): The name of the model. If not provided, it will be derived from the model_path.

        """
        super().__init__()
        model, processor, tokenizer, config = load_pretrained_model(
            model_path, model_base, model_name, device=device
        )

        self.level = level
        self.device = model.device
        self.dtype = model.dtype
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor
        self.model = model
        self.processor = processor
        self.cal_ids_ = [
            id_[0]
            for id_ in self.tokenizer(
                [" excellent", " good", " fair", " poor", " bad"]
            )["input_ids"]
        ]
        self.preferential_ids_ = [id_[0] for id_ in self.tokenizer(level)["input_ids"]]

        self.weight_tensor = (
            torch.Tensor([5, 4, 3, 2, 1]).to(self.dtype).to(self.device)
        )

    def forward(
        self,
        image_path: List[str],
        sys_prompt: str = "You are an expert in image quality assessment.",
    ):
        """
        Rates the quality of a list of input images, returning a score for each image between 1 and 5.

        Args:
            image_path (List[str]): The list of file paths for input images.

        Returns:
            Tuple[torch.Tensor, List[Dict[str, Any]]]:
                - A tensor containing the calculated score for each image, ranging from 1 to 5.
                - A list of dictionaries, each containing the filename and logits for the respective image.
        """
        if sys_prompt is not None:
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": "Can you rate the quality of the image in a single sentence?",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The quality of the image is"},
                    ],
                },
            ]
        else:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": "Can you rate the quality of the image in a single sentence?",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The quality of the image is"},
                    ],
                },
            ]
        prompt = self.processor.apply_chat_template(
            conversation, chat_template=llava_template, add_generation_prompt=False
        )
        print(prompt)
        prompts = [prompt] * len(image_path)
        with torch.inference_mode():  # 没有这一步会存储梯度图之类的导致OOM
            output_logits = []
            cal_logits = []
            for prompt, path in tqdm(zip(prompts, image_path), total=len(prompts)):
                print(path)
                inputs = self.processor(
                    images=load_image(path), text=prompt, return_tensors="pt"
                ).to(self.device, self.dtype)
                logit = self.model(**inputs)["logits"]
                output_logit = (
                    logit[:, -1, self.preferential_ids_]
                    .to(self.dtype)
                    .squeeze()
                    .tolist()
                )
                cal_logit = logit[:, -1, self.cal_ids_].to(self.dtype)
                cal_logits.append(cal_logit)
                logits_dict = defaultdict(
                    float, {level: val for level, val in zip(self.level, output_logit)}
                )
                print(cal_logit)
                # 构建结果字典
                output_logits.append(
                    {"filename": os.path.basename(path), "logits": logits_dict}
                )
                print(logits_dict)
            cal_logits = torch.stack(cal_logits, 0).squeeze()
            pred_mos_values = (
                torch.softmax(cal_logits, -1) @ self.weight_tensor
            ).tolist()
            if isinstance(pred_mos_values, float):
                pred_mos_values = [pred_mos_values]

            # 将pred_mos值加入到每个输出字典中
            for i, output in enumerate(output_logits):
                output["pred_mos"] = pred_mos_values[i]

            return output_logits


class QwenQAlignScorer(nn.Module):
    def __init__(
        self,
        model_path,
        model_base,
        device,
        level=[" excellent", " good", " fair", " poor", " bad"],
        model_name=None,
    ):
        """
        Initializes the QwenQAlignScorer class.

        Args:
            model_path (str): The path to the pretrained model.
            model_base (str): The base model to be used.
            device (torch.device): The device on which to load the model (e.g., 'cpu' or 'cuda' or 'cuda:0'). If device is "cuda", device_map will be "auto", otherwise, device_map will be device.
            level (List[str]): A list of strings representing the quality levels for scoring. Defaults to ["excellent", "good", "fair", "poor", "bad"].
            model_name (str, optional): The name of the model. If not provided, it will be derived from the model_path.

        """
        super().__init__()
        model, processor, tokenizer, _ = load_pretrained_model(
            model_path, model_base, model_name, device=device
        )

        self.tokenizer = tokenizer
        self.model = model
        self.level = level
        self.preferential_ids_ = [self.tokenizer.encode(text)[0] for text in level]
        self.cal_ids_ = [
            self.tokenizer.encode(text)[0]
            for text in [" excellent", " good", " fair", " poor", " bad"]
        ]
        self.dtype = model.dtype
        self.device = model.device
        self.weight_tensor = (
            torch.Tensor([5, 4, 3, 2, 1]).to(model.dtype).to(model.device)
        )

    def forward(
        self,
        image_path: List[str],
        sys_prompt: str = "You are an expert in image quality assessment.",
        bbox_list=None,
        dist_list=None,
    ):
        """
        Rates the quality of a list of input images, returning a score for each image between 1 and 5.

        Args:
            image_path (List[str]): The list of file paths for input images.

        Returns:
            Tuple[torch.Tensor, List[Dict[str, Any]]]:
                - A tensor containing the calculated score for each image, ranging from 1 to 5.
                - A list of dictionaries, each containing the filename and logits for the respective image.
        """
        prompts = []
        if bbox_list:
            prompts = [
                f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\nPicture: <img>{path}</img>\n{bbox}Accordingly, can you rate the quality of the image in a single sentence?<|im_end|>\n<|im_start|>assistant\nThe quality of the image is"
                for path, bbox in zip(image_path, bbox_list)
            ]
        elif dist_list:
            for path, dist in zip(image_path, dist_list):
                if "no distortion" in dist:
                    prompts.append(
                        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\nPicture: <img>{path}</img>\n{dist}Accordingly, can you rate the quality of the image in a single sentence?<|im_end|>\n<|im_start|>assistant\nThe quality of the image is"
                    )
                else:
                    prompts.append(
                        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\nPicture: <img>{path}</img>\nThe image contains one or more distortions as follows: {dist} Accordingly, can you rate the quality of the image in a single sentence?<|im_end|>\n<|im_start|>assistant\nThe quality of the image is"
                    )
        else:
            prompts = [
                f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\nPicture: <img>{path}</img>\nCan you rate the quality of the image in a single sentence?<|im_end|>\n<|im_start|>assistant\nThe quality of the image is"
                for path in image_path
            ]

        # print(prompts)
        with torch.inference_mode():
            output_logits = []
            cal_logits = []
            print(prompts[0])
            for prompt, path in tqdm(zip(prompts, image_path), total=len(prompts)):
                logit = self.model.forward_for_score(self.tokenizer, query=prompt)
                output_logit = logit[:, -1, self.preferential_ids_].squeeze().tolist()
                cal_logit = logit[:, -1, self.cal_ids_]
                cal_logits.append(cal_logit)
                logits_dict = defaultdict(
                    float, {level: val for level, val in zip(self.level, output_logit)}
                )

                # 构建结果字典
                output_logits.append(
                    {"filename": os.path.basename(path), "logits": logits_dict}
                )
            cal_logits = torch.stack(cal_logits, 0).squeeze()
            pred_mos_values = (
                torch.softmax(cal_logits, -1) @ self.weight_tensor
            ).tolist()
            if isinstance(pred_mos_values, float):
                pred_mos_values = [pred_mos_values]

            # 将pred_mos值加入到每个输出字典中
            for i, output in enumerate(output_logits):
                output["pred_mos"] = pred_mos_values[i]

            return output_logits


class Qwen2QAlignScorer(nn.Module):
    def __init__(
        self,
        model_path,
        model_base,
        device,
        level=[" excellent", " good", " fair", " poor", " bad"],
        model_name=None,
        use_custom_processor=True,
    ):
        """
        Initializes the LLaVAQAlignScorer class.

        Args:
            model_path (str): The path to the pretrained model.
            model_base (str): The base model to be used.
            device (torch.device): The device on which to load the model (e.g., 'cpu' or 'cuda' or 'cuda:0'), if device is "cuda", device_map will be "auto", otherwise, device_map will be device.
            level (List[str]): A list of strings representing the quality levels for scoring. Defaults to ["excellent", "good", "fair", "poor", "bad"].
            model_name (str, optional): The name of the model. If not provided, it will be derived from the model_path.

        """
        super().__init__()
        model, processor, tokenizer, config = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            device=device,
            use_custom_processor=use_custom_processor,
        )

        self.level = level
        self.device = model.device
        self.dtype = model.dtype
        self.tokenizer = processor.tokenizer
        self.model = model
        self.processor = processor
        self.processor.image_processor.max_pixels = 3072 * 28 * 28
        print(processor.image_processor.max_pixels)
        self.cal_ids_ = [
            id_[0]
            for id_ in self.tokenizer(
                [" excellent", " good", " fair", " poor", " bad"]
            )["input_ids"]
        ]
        self.preferential_ids_ = [id_[0] for id_ in self.tokenizer(level)["input_ids"]]

        self.weight_tensor = (
            torch.Tensor([5, 4, 3, 2, 1]).to(self.dtype).to(self.device)
        )

    def forward(
        self,
        image_path: List[str],
        sys_prompt: str = "You are an expert in image quality assessment.",
    ):
        """
        Rates the quality of a list of input images, returning a score for each image between 1 and 5.

        Args:
            image_path (List[str]): The list of file paths for input images.

        Returns:
            Tuple[torch.Tensor, List[Dict[str, Any]]]:
                - A tensor containing the calculated score for each image, ranging from 1 to 5.
                - A list of dictionaries, each containing the filename and logits for the respective image.
        """
        if sys_prompt is not None:
            prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Can you rate the quality of the image in a single sentence?<|im_end|>\n<|im_start|>assistant\nThe quality of the image is"
        else:
            prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Can you rate the quality of the image in a single sentence?<|im_end|>\n<|im_start|>assistant\nThe quality of the image is"
        prompts = [prompt] * len(image_path)
        with torch.inference_mode():  # 没有这一步会存储梯度图之类的导致OOM
            print("Inside inference mode:", torch.is_inference_mode_enabled())
            output_logits = []
            cal_logits = []
            for prompt, path in tqdm(zip(prompts, image_path), total=len(prompts)):
                # print(path)
                inputs = self.processor(
                    images=[load_image(path)], text=[prompt], return_tensors="pt"
                ).to(self.device, self.dtype)
                logit = self.model(**inputs)["logits"]
                output_logit = (
                    logit[:, -1, self.preferential_ids_]
                    .to(self.dtype)
                    .squeeze()
                    .tolist()
                )
                cal_logit = logit[:, -1, self.cal_ids_].to(self.dtype)
                print(cal_logit)
                cal_logits.append(cal_logit)
                logits_dict = defaultdict(
                    float, {level: val for level, val in zip(self.level, output_logit)}
                )
                # 构建结果字典
                output_logits.append(
                    {"filename": os.path.basename(path), "logits": logits_dict}
                )
                # print(logits_dict)
            cal_logits = torch.stack(cal_logits, 0).squeeze()
            pred_mos_values = (
                torch.softmax(cal_logits, -1) @ self.weight_tensor
            ).tolist()
            if isinstance(pred_mos_values, float):
                pred_mos_values = [pred_mos_values]

            # 将pred_mos值加入到每个输出字典中
            for i, output in enumerate(output_logits):
                output["pred_mos"] = pred_mos_values[i]

            return output_logits


class InternVL2QAlignScorer(nn.Module):
    def __init__(
        self,
        model_path,
        model_base,
        level=["excellent", "good", "fair", "poor", "bad"],
        model_name=None,
    ):
        """
        Initializes the LLaVAQAlignScorer class.

        Args:
            model_path (str): The path to the pretrained model.
            model_base (str): The base model to be used.
            device (torch.device): The device on which to load the model (e.g., 'cpu' or 'cuda' or 'cuda:0'), if device is "cuda", device_map will be "auto", otherwise, device_map will be device.
            level (List[str]): A list of strings representing the quality levels for scoring. Defaults to ["excellent", "good", "fair", "poor", "bad"].
            model_name (str, optional): The name of the model. If not provided, it will be derived from the model_path.

        """
        super().__init__()
        model, processor, tokenizer, config = load_pretrained_model(
            model_path,
            model_base,
            model_name,
        )
        self.level = level
        self.device = model.device
        self.dtype = model.dtype
        self.tokenizer = tokenizer
        self.model = model
        self.max_num = config.max_dynamic_patch
        self.cal_ids_ = [
            id_[1]
            for id_ in self.tokenizer(
                [" excellent", " good", " fair", " poor", " bad"]
            )["input_ids"]
        ]
        # id_为[1, level_id]
        self.preferential_ids_ = [id_[1] for id_ in self.tokenizer(level)["input_ids"]]

        self.weight_tensor = (
            torch.Tensor([5, 4, 3, 2, 1]).to(self.dtype).to(self.device)
        )

    def forward(
        self,
        image_path: List[str],
        sys_prompt: str = "You are an expert in image quality assessment.",
    ):
        """
        Rates the quality of a list of input images, returning a score for each image between 1 and 5.

        Args:
            image_path (List[str]): The list of file paths for input images.

        Returns:
            Tuple[torch.Tensor, List[Dict[str, Any]]]:
                - A tensor containing the calculated score for each image, ranging from 1 to 5.
                - A list of dictionaries, each containing the filename and logits for the respective image.
        """
        prompts = [
            "Can you rate the quality of the image in a single sentence?",
        ] * len(image_path)
        with torch.inference_mode():  # 没有这一步会存储梯度图之类的导致OOM
            output_logits = []
            cal_logits = []
            for prompt, path in tqdm(zip(prompts, image_path), total=len(prompts)):
                print(path)
                pixel_values = (
                    internvl2_load_image(path, max_num=self.max_num)
                    .to(self.dtype)
                    .to(self.device)
                )
                logit = self.model.get_logits_for_image_score(
                    self.tokenizer, pixel_values, prompt
                )
                output_logit = (
                    logit[:, -1, self.preferential_ids_]
                    .to(self.dtype)
                    .squeeze()
                    .tolist()
                )
                cal_logit = logit[:, -1, self.cal_ids_].to(self.dtype)
                cal_logits.append(cal_logit)
                logits_dict = defaultdict(
                    float, {level: val for level, val in zip(self.level, output_logit)}
                )
                print(cal_logit)
                # 构建结果字典
                output_logits.append(
                    {"filename": os.path.basename(path), "logits": logits_dict}
                )
                print(logits_dict)
            cal_logits = torch.stack(cal_logits, 0).squeeze()
            pred_mos_values = (
                torch.softmax(cal_logits, -1) @ self.weight_tensor
            ).tolist()
            if isinstance(pred_mos_values, float):
                pred_mos_values = [pred_mos_values]

            # 将pred_mos值加入到每个输出字典中
            for i, output in enumerate(output_logits):
                output["pred_mos"] = pred_mos_values[i]

            return output_logits


if __name__ == "__main__":
    from transformers import AutoProcessor
    # try:
    #     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    #     debugpy.listen(("localhost", 9501))
    #     print("Waiting for debugger attach")
    #     debugpy.wait_for_client()
    # except Exception:
    #     pass

    model_path = "../models/qwen2-vl-7b-instruct"
    model, processor, tokenizer, config = load_pretrained_model(
        model_path, device="cuda:1"
    )
    image_path = "../gen_prompt/dataset/ref_image/bad.png"
    image = load_image(image_path)
    processor = AutoProcessor.from_pretrained(model_path)
    text_prompt = "<|im_start|>system\nYou are an expert in image quality assessment.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Can you rate the quality of the image in a single sentence?<|im_end|>\n<|im_start|>assistant\nThe quality of the image is"
    with torch.inference_mode():  # 没有这一步会存储梯度图之类的导致OOM
        inputs = processor(images=[image], text=[text_prompt], return_tensors="pt").to(
            "cuda:1", torch.bfloat16
        )
        output1 = model(**inputs)["logits"]
    print(output1.shape)
