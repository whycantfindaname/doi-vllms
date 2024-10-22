import os
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
from builder import load_image, load_pretrained_model
from PIL import Image
from tqdm import tqdm


def expand2square(pil_img, background_color):
    """
    Expands a non-square image into a square image, padding the shorter side with the specified background color.

    Args:
    pil_img (PIL.Image.Image): The input image object.
    background_color (tuple): The background color to use for padding, typically the image's average color.

    Returns:
    PIL.Image.Image: The padded square image.
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


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
            device (torch.device): The device on which to load the model (e.g., 'cpu' or 'cuda').
            level (List[str]): A list of strings representing the quality levels for scoring. Defaults to ["excellent", "good", "fair", "poor", "bad"].
            model_name (str, optional): The name of the model. If not provided, it will be derived from the model_path.

        """
        super().__init__()
        if model_name is None:
            model_name = os.path.basename(model_path)
        processor, model, _ = load_pretrained_model(
            model_path, model_base, model_name, device=device
        )

        self.level = level

        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor
        self.model = model
        self.processor = processor

        self.prompt = "<image>\nUSER: Can you evaluate the quality of the image in a single sentence?\nASSISTANT: The quality of the image is"

        self.cal_ids_ = [
            id_[0]
            for id_ in self.tokenizer(
                [" excellent", " good", " fair", " poor", " bad"]
            )["input_ids"]
        ]
        self.preferential_ids_ = [id_[0] for id_ in self.tokenizer(level)["input_ids"]]

        self.weight_tensor = torch.Tensor([5, 4, 3, 2, 1]).half().to(model.device)

    def forward(self, image_path: List[str], temperature=1):
        """
        Rates the quality of a list of input images, returning a score for each image between 1 and 5.

        Args:
            image_path (List[str]): The list of file paths for input images.

        Returns:
            Tuple[torch.Tensor, List[Dict[str, Any]]]:
                - A tensor containing the calculated score for each image, ranging from 1 to 5.
                - A list of dictionaries, each containing the filename and logits for the respective image.
        """
        # Expand each input image to a square and convert it to a tensor format suitable for the model
        image = [
            expand2square(
                load_image(img),
                tuple(int(x * 255) for x in self.image_processor.image_mean),
            )
            for img in image_path
        ]

        with torch.inference_mode():
            inputs = self.processor(
                [self.prompt] * len(image), images=image, return_tensors="pt"
            ).to(self.model.device)
            inputs["pixel_values"] = inputs["pixel_values"].half()

            cal_logits = (
                self.model(**inputs).logits[:, -1, self.cal_ids_].half() / temperature
            )
            output_logits = (
                (self.model(**inputs).logits[:, -1, self.preferential_ids_].half())
                .squeeze()
                .tolist()
            ) / temperature

            pred_mos = (torch.softmax(cal_logits, -1) @ self.weight_tensor).tolist()
            logits_dicts = []
            for i, logits in tqdm(enumerate(output_logits)):
                logits_dict = defaultdict(
                    float, {level: logits[j] for j, level in enumerate(self.level)}
                )
                logits_dicts.append(
                    {
                        "filename": os.path.basename(image_path[i]),
                        "logits": logits_dict,
                        "pred_mos": pred_mos[i],
                    }
                )

            return logits_dicts


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
            device (torch.device): The device on which to load the model (e.g., 'cpu' or 'cuda').
            level (List[str]): A list of strings representing the quality levels for scoring. Defaults to ["excellent", "good", "fair", "poor", "bad"].
            model_name (str, optional): The name of the model. If not provided, it will be derived from the model_path.

        """
        super().__init__()
        if model_name is None:
            model_name = os.path.basename(model_path)
        tokenizer, model, _ = load_pretrained_model(
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
        self.weight_tensor = torch.Tensor([5, 4, 3, 2, 1]).half().to(model.device)

    def forward(self, image_path: List[str], temperature=1, sys_prompt: str = "You are an expert in image quality assessment. Your task is to evaluate the quality of an image.", bbox_list=None, dist_list=None):
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
                f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\nPicture: <img>{path}</img>\n{bbox}Accordingly, can you evaluate the quality of the image in a single sentence?<|im_end|>\n<|im_start|>assistant\nThe quality of the image is"
                for path, bbox in zip(image_path, bbox_list)
            ]
        elif dist_list:
            for path, dist in zip(image_path, dist_list):
                if "no distortion" in dist:
                    prompts.append(
                        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\nPicture: <img>{path}</img>\n{dist}Accordingly, can you evaluate the quality of the image in a single sentence?<|im_end|>\n<|im_start|>assistant\nThe quality of the image is"
                    )
                else:
                    prompts.append(
                        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\nPicture: <img>{path}</img>\nThe image contains one or more distortions as follows: {dist} Accordingly, can you evaluate the quality of the image in a single sentence?<|im_end|>\n<|im_start|>assistant\nThe quality of the image is"
                    )
        else:
            prompts = [
                f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\nPicture: <img>{path}</img>\nCan you evaluate the quality of the image in a single sentence?<|im_end|>\n<|im_start|>assistant\nThe quality of the image is"
                for path in image_path
            ]

        # print(prompts)
        with torch.inference_mode():
            output_logits = []
            cal_logits = []
            print(prompts[0])
            for prompt, path in tqdm(zip(prompts, image_path), total=len(prompts)):
                logit = self.model.forward_for_score(self.tokenizer, query=prompt)
                output_logit = (
                    (logit[:, -1, self.preferential_ids_].half() / temperature)
                    .squeeze()
                    .tolist()
                )
                cal_logit = logit[:, -1, self.cal_ids_].half() / temperature
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

            # 将pred_mos值加入到每个输出字典中
            for i, output in enumerate(output_logits):
                output["pred_mos"] = pred_mos_values[i]

            return output_logits


