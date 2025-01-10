      
import os
from collections import defaultdict

import torch
import torch.nn as nn
from builder import (
    internvl_load_image,
    load_image,
    load_pretrained_model,
)
from tqdm import tqdm

from typing import List, Union


def find_pattern_next_position(pattern_ids: List, generated_ids_trimmed: Union[List, torch.Tensor], device):
    # Ensure pattern_ids is a tensor, convert it if it is a list
    if isinstance(pattern_ids, list):
        pattern_ids = torch.tensor(pattern_ids, device=device)

    # Ensure pattern_ids and generated_ids_trimmed are tensors
    if not isinstance(pattern_ids, torch.Tensor) or not isinstance(generated_ids_trimmed, torch.Tensor):
        raise ValueError("Both pattern_ids and generated_ids_trimmed must be torch tensors.")

    # Get the lengths of pattern_ids and generated_ids_trimmed
    pattern_len = len(pattern_ids)
    generated_len = len(generated_ids_trimmed)

    # Iterate over generated_ids_trimmed in reverse order to find the last matching sequence
    for i in range(generated_len - pattern_len, -1, -1):  # Loop from end to start
        # Check if the sub-sequence matches pattern_ids
        if torch.equal(generated_ids_trimmed[i:i + pattern_len], pattern_ids):
            # Return the position of the next element after the match
            return i + pattern_len  # The position of the next element
    
    # If no match is found, return False
    return None

# 防止匹配不到
def find_pattern_last_position(pattern_ids: List, generated_ids_trimmed: Union[List, torch.Tensor], device):
    # Ensure pattern_ids and generated_ids_trimmed are tensors
    if not isinstance(pattern_ids, torch.Tensor) or not isinstance(generated_ids_trimmed, torch.Tensor):
        pattern_ids = torch.tensor(pattern_ids, device=device)
        generated_ids_trimmed = torch.tensor(generated_ids_trimmed, device=device)

    # Get the lengths of pattern_ids and generated_ids_trimmed
    pattern_len = len(pattern_ids)
    generated_len = len(generated_ids_trimmed)

    # Iterate over generated_ids_trimmed in reverse order to find the last matching sequence
    for i in range(generated_len - pattern_len, -1, -1):  # Loop from end to start
        # Check if the sub-sequence matches pattern_ids
        if torch.equal(generated_ids_trimmed[i:i + pattern_len], pattern_ids):
            # Return the position of the last matching element
            return i  # The position where the pattern starts
    
    # If no match is found, return False
    return None

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

class LLaVACOTScorer(nn.Module):
    def __init__(
        self,
        model_path,
        model_base,
        device,
        level=[" excellent", " good", " fair", " poor", " bad"],
        model_name=None
    ):
        """

        Args:
            model_path (str): The path to the pretrained model.
            model_base (str): The base model to be used.
            device (torch.device): The device on which to load the model (e.g., 'cpu' or 'cuda' or 'cuda:0'). If device is "cuda", device_map will be "auto", otherwise, device_map will be device.
            level (List[str]): A list of strings representing the quality levels for scoring. Defaults to ["excellent", "good", "fair", "poor", "bad"].
            model_name (str, optional): The name of the model. If not provided, it will be derived from the model_path.

        """
        super().__init__()
        model, processor, tokenizer, _ = load_pretrained_model(
            model_path, model_base, model_name, device=device, torch_dtype=torch.float16
        )
        patterns = [" Thus, the quality of the image is", "Thus, the quality of the image is", " The quality of the image is", " the quality of the image is", " The quality of this image is", " the quality of this image is", " The overall image quality is", " the overall image quality is", " The overall quality of the image is"]

        patterns_ids = []
        for pattern in patterns:
            patterns_ids.append(tokenizer.encode(pattern))
        
        level = [" excellent", " good", " fair", " poor", " bad"]
        levels_ids = []
        for level_item in level:
            levels_ids.append(tokenizer.encode(level_item))
        
        self.levels_ids = levels_ids
        self.patterns_ids = patterns_ids
        self.level = level
        self.tokenizer = tokenizer
        self.model = model  
        self.processor = processor
        self.cal_ids_ = [
            id_[0]
            for id_ in self.tokenizer(
                [" excellent", " good", " fair", " poor", " bad"]
            )["input_ids"]
        ]
        self.preferential_ids_ = [id_[0] for id_ in self.tokenizer(level)["input_ids"]]
        self.dtype = model.dtype
        self.device = model.device
        self.weight_tensor = (
            torch.Tensor([5, 4, 3, 2, 1]).to(model.device).to(torch.float32)
        )
    def forward(
        self,
        image_path: List[str],
        sys_prompt: str = "You are a helpful assistant.",
        query: str = "You are an expert in image quality assessment. Your task is to assess the overall quality of the provided image.\nTo assess the image quality, you should think step by step.\nFirst step, provide a brief description of the image content.\nSecond step, analyze the overall image quality and visual perception.\n- If there is no distortion present in the image, focus solely on what you observe in the image and describe the image's visual aspects, such as visibility, detail discernment, clarity, brightness, lighting, composition, and texture.\n- If distortions are present, identify the distortions and briefly analyze each occurrence of every distortion type.Explain how each distortion affects the visual appearance and perception of specific objects or regions in the image.\nThird step, If distortions are present, identify the key distortions that have the most significant impact on the overall image quality.Provide detailed reasoning about how these key distortions affect the image's visual perception, especially regarding sharpness, clarity, and detail. Combine the analysis of key degradations and low-level attributes into a cohesive paragraph.\nFInal step, conclude your answer with this sentence: 'Thus, the quality of the image is (one of the following five quality levels: bad, poor, fair, good, excellent)'."
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
        conversation = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": query,
                        },
                    ],
                }
            ]
        prompt = self.processor.apply_chat_template(
            conversation, chat_template=llava_template, add_generation_prompt=True
        )
        prompts = [prompt] * len(image_path)
        greedy_decoding_config = dict(do_sample=False,num_beams=1,top_p=None,top_k=None, max_new_tokens=512, return_dict_in_generate=True, output_logits=True, temperature=None)
        # print(prompts[0])
        with torch.inference_mode():
            output_logits = []
            cal_logits = []
            print(prompts[0])
            for prompt, path in tqdm(zip(prompts, image_path), total=len(prompts)):
                inputs = self.processor(
                    images=[load_image(path)], text=[prompt], return_tensors="pt", padding=True
                ).to(self.device, self.dtype)
                outputs = self.model.generate(
                        **inputs,
                        **greedy_decoding_config,
                    )
                logit = outputs.logits
                seq = outputs.sequences
                generated_ids_trimmed = [
                    out_ids[len(in_ids) : ] for in_ids, out_ids in zip(inputs.input_ids, seq)
                ]
                answer = self.tokenizer.batch_decode(generated_ids_trimmed)
                assert len(logit) == len(generated_ids_trimmed[0]), f'logit and generated_ids_trimmed should have same length, but logit has length {len(logit)} and generated_ids_trimmed has length {len(generated_ids_trimmed[0])}'
                
                final_token_pos = None  # Initialize as None instead of -1

                # Try finding pattern in patterns_ids
                for pattern_ids in self.patterns_ids:
                    final_token_pos = find_pattern_next_position(pattern_ids, generated_ids_trimmed[0], self.device)
                    if final_token_pos is not None:
                        break
                    else:
                        print(f'Pattern {self.tokenizer.decode(pattern_ids)} not found in generated_ids_trimmed')

                # If a pattern was found, check if it's in the allowed ids
                if final_token_pos is not None:  # Ensure that a valid position is returned
                    final_token = generated_ids_trimmed[0][final_token_pos]
                    final_word = self.tokenizer.decode(final_token)
                    print(f"Found token at position {final_token_pos}: {final_word}")
                    
                    if final_token not in self.cal_ids_:
                        print(f"Token {final_word} not in cal_ids_, resetting final_token_pos.")
                        final_token_pos = None  # Reset if not in cal_ids_

                # If no match found in the next position search, check the last position in levels_ids
                if final_token_pos is None:  # Check if no pattern was found
                    for level_id in self.levels_ids:
                        final_token_pos = find_pattern_last_position(level_id, generated_ids_trimmed[0], self.device)
                        if final_token_pos is not None:
                            break
                        else:
                            print(f'Level {self.tokenizer.decode(level_id)} not found in generated_ids_trimmed')

                # Final check: If a valid position is found, decode and print the token
                if final_token_pos is not None:  # Ensure that a valid position is returned
                    final_token = generated_ids_trimmed[0][final_token_pos]
                    final_word = self.tokenizer.decode(final_token)
                    print(f"Found token at position {final_token_pos}: {final_word}")
                else:
                    print("No valid pattern found in the generated sequence.")
                    continue  # Skip to the next iteration if no valid pattern is found
                                
                # assert final_token in self.cal_ids_, f'final token is not in [" excellent", " good", " fair", " poor", " bad"], but is {final_word}'
                cal_logit = logit[final_token_pos][:, self.cal_ids_]
                output_logit = logit[final_token_pos][:, self.preferential_ids_].squeeze().tolist()
                print(cal_logit)
                print(answer)
                cal_logits.append(cal_logit)
                logits_dict = defaultdict(
                    float, {level: val for level, val in zip(self.level, output_logit)}
                )

                # 构建结果字典
                output_logits.append(
                    {"filename": os.path.basename(path), "logits": logits_dict, "answer": answer}
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
            model_path, model_base, model_name, device=device, torch_dtype=torch.float16
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
        query: str = 'Can you rate the quality of the image in a single sentence?'
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
            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\nPicture: <img>{path}</img>\n{query}<|im_end|>\n<|im_start|>assistant\nThe quality of the image is"
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
                print(cal_logit)
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

class QwenCOTScorer(nn.Module):
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
            model_path, model_base, model_name, device=device, torch_dtype=torch.float16
        )
        patterns = [" Thus, the quality of the image is", "Thus, the quality of the image is", " The quality of the image is", " the quality of the image is", " The overall image quality is", " the overall image quality is"]

        patterns_ids = []
        for pattern in patterns:
            patterns_ids.append(tokenizer.encode(pattern))
        
        self.patterns_ids = patterns_ids
        self.level = level
        self.tokenizer = tokenizer
        self.model = model  
        self.preferential_ids_ = [self.tokenizer.encode(text)[0] for text in level]
        self.cal_ids_ = [
            self.tokenizer.encode(text)[0]
            for text in [" excellent", " good", " fair", " poor", " bad"]
        ]
        self.dtype = model.dtype
        self.device = model.device
        self.weight_tensor = (
            torch.Tensor([5, 4, 3, 2, 1]).to(model.device).to(torch.float32)
        )
        self.stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    def forward(
        self,
        image_path: List[str],
        sys_prompt: str = "You are a helpful assistant.",
        query: str = "You are an expert in image quality assessment. Your task is to assess the overall quality of the provided image.\nTo assess the image quality, you should think step by step.\nFirst step, provide a brief description of the image content.\nSecond step, analyze the overall image quality and visual perception.\n- If there is no distortion present in the image, focus solely on what you observe in the image and describe the image's visual aspects, such as visibility, detail discernment, clarity, brightness, lighting, composition, and texture.\n- If distortions are present, identify the distortions and briefly analyze each occurrence of every distortion type.Explain how each distortion affects the visual appearance and perception of specific objects or regions in the image.\nThird step, If distortions are present, identify the key distortions that have the most significant impact on the overall image quality.Provide detailed reasoning about how these key distortions affect the image's visual perception, especially regarding sharpness, clarity, and detail. Combine the analysis of key degradations and low-level attributes into a cohesive paragraph.\nFInal step, conclude your answer with this sentence: 'Thus, the quality of the image is (one of the following five quality levels: bad, poor, fair, good, excellent)'."
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
            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\nPicture: <img>{path}</img>\n{query}<|im_end|>\n<|im_start|>assistant\n"
            for path in image_path
        ]
        greedy_decoding_config = dict(do_sample=False,num_beams=1,top_p=None,top_k=None)
        # print(prompts)
        with torch.inference_mode():
            output_logits = []
            cal_logits = []
            print(prompts[0])
            for prompt, path in tqdm(zip(prompts, image_path), total=len(prompts)):
                context_tokens = self.tokenizer.encode(prompt)
                input_ids = torch.tensor([context_tokens]).to(self.model.device)
                outputs = self.model.generate(
                        input_ids,
                        stop_words_ids=self.stop_words_ids,
                        return_dict_in_generate=True,
                        output_logits=True,
                        **greedy_decoding_config,
                    )
                logit = outputs.logits
                seq = outputs.sequences
                generated_ids_trimmed = [
                    out_ids[len(in_ids) : ] for in_ids, out_ids in zip(input_ids, seq)
                ]
                assert len(logit) == len(generated_ids_trimmed[0]), f'logit and generated_ids_trimmed should have same length, but logit has length {len(logit)} and generated_ids_trimmed has length {len(generated_ids_trimmed[0])}'
                for pattern_ids in self.patterns_ids:
                    final_token_pos = find_pattern_next_position(pattern_ids, generated_ids_trimmed[0], self.device)
                    if final_token_pos:
                        break
                    else:
                        print(f'pattern {self.tokenizer.decode(pattern_ids)} not found in generated_ids_trimmed')
                        final_token_pos = -1
                        
                final_token = generated_ids_trimmed[0][final_token_pos]
                final_word = self.tokenizer.decode(final_token)
                if final_token not in self.cal_ids_:
                    print(f'final token is not in [" excellent", " good", " fair", " poor", " bad"], but is {final_word}')
                    print(self.tokenizer.batch_decode(generated_ids_trimmed))
                    continue
                # assert final_token in self.cal_ids_, f'final token is not in [" excellent", " good", " fair", " poor", " bad"], but is {final_word}'
                cal_logit = logit[final_token_pos][:, self.cal_ids_]
                output_logit = logit[final_token_pos][:, self.preferential_ids_].squeeze().tolist()
                print(cal_logit)
                print(self.tokenizer.batch_decode(generated_ids_trimmed))
                cal_logits.append(cal_logit)
                logits_dict = defaultdict(
                    float, {level: val for level, val in zip(self.level, output_logit)}
                )

                # 构建结果字典
                output_logits.append(
                    {"filename": os.path.basename(path), "logits": logits_dict, "answer": self.tokenizer.batch_decode(generated_ids_trimmed)}
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
        # self.processor.image_processor.max_pixels = 3072 * 28 * 28
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

class Qwen2COTScorer(nn.Module):
    def __init__(
        self,
        model_path,
        model_base,
        device,
        level=[" excellent", " good", " fair", " poor", " bad"],
        model_name=None,
        use_custom_processor=False,
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
            model_path, model_base, model_name, device=device, torch_dtype=torch.float16, use_custom_processor=use_custom_processor
        )
        patterns = [" Thus, the quality of the image is", "Thus, the quality of the image is", " The quality of the image is", " the quality of the image is", " The quality of this image is", " the quality of this image is", " The overall image quality is", " the overall image quality is", " The overall quality of the image is"]

        patterns_ids = []
        for pattern in patterns:
            patterns_ids.append(tokenizer.encode(pattern))
        
        level = [" excellent", " good", " fair", " poor", " bad"]
        levels_ids = []
        for level_item in level:
            levels_ids.append(tokenizer.encode(level_item))
        
        self.levels_ids = levels_ids
        self.patterns_ids = patterns_ids
        self.level = level
        self.tokenizer = tokenizer
        self.model = model  
        self.processor = processor
        print(processor.image_processor.max_pixels)
        self.cal_ids_ = [
            id_[0]
            for id_ in self.tokenizer(
                [" excellent", " good", " fair", " poor", " bad"]
            )["input_ids"]
        ]
        self.preferential_ids_ = [id_[0] for id_ in self.tokenizer(level)["input_ids"]]
        self.dtype = model.dtype
        self.device = model.device
        self.weight_tensor = (
            torch.Tensor([5, 4, 3, 2, 1]).to(model.device).to(torch.float32)
        )
    def forward(
        self,
        image_path: List[str],
        sys_prompt: str = "You are a helpful assistant.",
        query: str = "You are an expert in image quality assessment. Your task is to assess the overall quality of the provided image.\nTo assess the image quality, you should think step by step.\nFirst step, provide a brief description of the image content.\nSecond step, analyze the overall image quality and visual perception.\n- If there is no distortion present in the image, focus solely on what you observe in the image and describe the image's visual aspects, such as visibility, detail discernment, clarity, brightness, lighting, composition, and texture.\n- If distortions are present, identify the distortions and briefly analyze each occurrence of every distortion type.Explain how each distortion affects the visual appearance and perception of specific objects or regions in the image.\nThird step, If distortions are present, identify the key distortions that have the most significant impact on the overall image quality.Provide detailed reasoning about how these key distortions affect the image's visual perception, especially regarding sharpness, clarity, and detail. Combine the analysis of key degradations and low-level attributes into a cohesive paragraph.\nFInal step, conclude your answer with this sentence: 'Thus, the quality of the image is (one of the following five quality levels: bad, poor, fair, good, excellent)'."
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
            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{query}<|im_end|>\n<|im_start|>assistant\n"
        ] * len(image_path)
        greedy_decoding_config = dict(do_sample=False,num_beams=1,top_p=None,top_k=None, max_new_tokens=512, return_dict_in_generate=True, output_logits=True, temperature=None)
        # print(prompts[0])
        with torch.inference_mode():
            output_logits = []
            cal_logits = []
            print(prompts[0])
            for prompt, path in tqdm(zip(prompts, image_path), total=len(prompts)):
                inputs = self.processor(
                    images=[load_image(path)], text=[prompt], return_tensors="pt", padding=True
                ).to(self.device, self.dtype)
                outputs = self.model.generate(
                        **inputs,
                        **greedy_decoding_config,
                    )
                logit = outputs.logits
                seq = outputs.sequences
                generated_ids_trimmed = [
                    out_ids[len(in_ids) : ] for in_ids, out_ids in zip(inputs.input_ids, seq)
                ]
                answer = self.tokenizer.batch_decode(generated_ids_trimmed)
                assert len(logit) == len(generated_ids_trimmed[0]), f'logit and generated_ids_trimmed should have same length, but logit has length {len(logit)} and generated_ids_trimmed has length {len(generated_ids_trimmed[0])}'
                
                final_token_pos = None  # Initialize as None instead of -1

                # Try finding pattern in patterns_ids
                for pattern_ids in self.patterns_ids:
                    final_token_pos = find_pattern_next_position(pattern_ids, generated_ids_trimmed[0], self.device)
                    if final_token_pos is not None:
                        break
                    else:
                        print(f'Pattern {self.tokenizer.decode(pattern_ids)} not found in generated_ids_trimmed')

                # If a pattern was found, check if it's in the allowed ids
                if final_token_pos is not None:  # Ensure that a valid position is returned
                    final_token = generated_ids_trimmed[0][final_token_pos]
                    final_word = self.tokenizer.decode(final_token)
                    print(f"Found token at position {final_token_pos}: {final_word}")
                    
                    if final_token not in self.cal_ids_:
                        print(f"Token {final_word} not in cal_ids_, resetting final_token_pos.")
                        final_token_pos = None  # Reset if not in cal_ids_

                # If no match found in the next position search, check the last position in levels_ids
                if final_token_pos is None:  # Check if no pattern was found
                    for level_id in self.levels_ids:
                        final_token_pos = find_pattern_last_position(level_id, generated_ids_trimmed[0], self.device)
                        if final_token_pos is not None:
                            break
                        else:
                            print(f'Level {self.tokenizer.decode(level_id)} not found in generated_ids_trimmed')

                # Final check: If a valid position is found, decode and print the token
                if final_token_pos is not None:  # Ensure that a valid position is returned
                    final_token = generated_ids_trimmed[0][final_token_pos]
                    final_word = self.tokenizer.decode(final_token)
                    print(f"Found token at position {final_token_pos}: {final_word}")
                else:
                    print("No valid pattern found in the generated sequence.")
                    continue  # Skip to the next iteration if no valid pattern is found
                                
                # assert final_token in self.cal_ids_, f'final token is not in [" excellent", " good", " fair", " poor", " bad"], but is {final_word}'
                cal_logit = logit[final_token_pos][:, self.cal_ids_]
                output_logit = logit[final_token_pos][:, self.preferential_ids_].squeeze().tolist()
                print(cal_logit)
                print(answer)
                cal_logits.append(cal_logit)
                logits_dict = defaultdict(
                    float, {level: val for level, val in zip(self.level, output_logit)}
                )

                # 构建结果字典
                output_logits.append(
                    {"filename": os.path.basename(path), "logits": logits_dict, "answer": answer}
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

class InternVLQAlignScorer(nn.Module):
    def __init__(
        self,
        model_path,
        model_base,
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
        model_name = os.path.basename(model_path)
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
            "You are an expert in imaage quality assessment. Can you rate the quality of the image in a single sentence?",
        ] * len(image_path)
        with torch.inference_mode():  # 没有这一步会存储梯度图之类的导致OOM
            output_logits = []
            cal_logits = []
            for prompt, path in tqdm(zip(prompts, image_path), total=len(prompts)):
                print(path)
                pixel_values = (
                    internvl_load_image(path, max_num=self.max_num)
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

class InternVLCOTScorer(nn.Module):
    def __init__(
        self,
        model_path,
        model_base,
        device,
        model_name=None
    ):
        """

        Args:
            model_path (str): The path to the pretrained model.
            model_base (str): The base model to be used.
            device (torch.device): The device on which to load the model (e.g., 'cpu' or 'cuda' or 'cuda:0'). If device is "cuda", device_map will be "auto", otherwise, device_map will be device.
            level (List[str]): A list of strings representing the quality levels for scoring. Defaults to ["excellent", "good", "fair", "poor", "bad"].
            model_name (str, optional): The name of the model. If not provided, it will be derived from the model_path.

        """
        super().__init__()
        model, processor, tokenizer, config = load_pretrained_model(
            model_path, model_base, model_name, device=device, torch_dtype=torch.float16
        )

        patterns = [" Thus, the quality of the image is", "Thus, the quality of the image is", " The quality of the image is", " the quality of the image is", " The quality of this image is", " the quality of this image is", " The overall image quality is", " the overall image quality is", " The overall quality of the image is"]

        patterns_ids = []
        for pattern in patterns:
            patterns_ids.append(tokenizer.encode(pattern)[1:])
        # ' excellent' as [1, 9202]
        # ' good' as [1, 1811]
        # ' fair as [1, 6776]
        # ' poor' as [1, 7989]
        # ' bad' as [1, 4028]
        level = [" excellent", " good", " fair", " poor", " bad"]
        levels_ids = []
        for level_item in level:
            levels_ids.append(tokenizer.encode(level_item)[1])
        
        self.levels_ids = levels_ids
        self.patterns_ids = patterns_ids
        self.level = level
        self.tokenizer = tokenizer
        self.max_num = config.max_dynamic_patch
        self.model = model  
        self.processor = processor
        self.cal_ids_ = [
            id_[1]
            for id_ in self.tokenizer(
                level
            )["input_ids"]
        ]
        self.dtype = model.dtype
        self.device = model.device
        self.weight_tensor = (
            torch.Tensor([5, 4, 3, 2, 1]).to(model.device).to(torch.float32)
        )
    def forward(
        self,
        image_path: List[str],
        sys_prompt: str = None,
        query: str = "You are an expert in image quality assessment. Your task is to assess the overall quality of the provided image.\nTo assess the image quality, you should think step by step.\nFirst step, provide a brief description of the image content.\nSecond step, analyze the overall image quality and visual perception.\n- If there is no distortion present in the image, focus solely on what you observe in the image and describe the image's visual aspects, such as visibility, detail discernment, clarity, brightness, lighting, composition, and texture.\n- If distortions are present, identify the distortions and briefly analyze each occurrence of every distortion type.Explain how each distortion affects the visual appearance and perception of specific objects or regions in the image.\nThird step, If distortions are present, identify the key distortions that have the most significant impact on the overall image quality.Provide detailed reasoning about how these key distortions affect the image's visual perception, especially regarding sharpness, clarity, and detail. Combine the analysis of key degradations and low-level attributes into a cohesive paragraph.\nFInal step, conclude your answer with this sentence: 'Thus, the quality of the image is (one of the following five quality levels: bad, poor, fair, good, excellent)'."
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
        greedy_decoding_config = dict(do_sample=False,num_beams=1,top_p=None,top_k=None, max_new_tokens=512, return_dict_in_generate=True, output_logits=True, temperature=None)
        with torch.inference_mode():
            output_logits = []
            cal_logits = []
            for path in tqdm(image_path, total=len(image_path)):
                pixel_values = (
                    internvl_load_image(path, max_num=self.max_num)
                    .to(self.dtype)
                    .to(self.device)
                )
                outputs = self.model.generate_for_cot_score(
                        tokenizer=self.tokenizer,
                        pixel_values=pixel_values,
                        question=query,
                        sys=sys_prompt,
                        cot_generation_config = greedy_decoding_config,
                    )
                logit = outputs.logits
                seq = outputs.sequences
                generated_ids_trimmed = seq
                answer = self.tokenizer.batch_decode(generated_ids_trimmed)
                print(answer)
                assert len(logit) == len(generated_ids_trimmed[0]), f'logit and generated_ids_trimmed should have same length, but logit has length {len(logit)} and generated_ids_trimmed has length {len(generated_ids_trimmed[0])}'
                
                final_token_pos = None  # Initialize as None instead of -1

                # Try finding pattern in patterns_ids
                for pattern_ids in self.patterns_ids:
                    final_token_pos = find_pattern_next_position(pattern_ids, generated_ids_trimmed[0], self.device)
                    if final_token_pos is not None:
                        break
                    else:
                        print(f'Pattern {self.tokenizer.decode(pattern_ids)} not found in generated_ids_trimmed')

                # If a pattern was found, check if it's in the allowed ids
                if final_token_pos is not None:  # Ensure that a valid position is returned
                    final_token = generated_ids_trimmed[0][final_token_pos]
                    final_word = self.tokenizer.decode(final_token)
                    print(f"Found token at position {final_token_pos}: {final_word}")
                    
                    if final_token not in self.cal_ids_:
                        print(f"Token {final_word} not in cal_ids_, resetting final_token_pos.")
                        final_token_pos = None  # Reset if not in cal_ids_

                # If no match found in the next position search, check the last position in levels_ids
                if final_token_pos is None:  # Check if no pattern was found
                    for level_id in self.levels_ids:
                        final_token_pos = find_pattern_last_position(level_id, generated_ids_trimmed[0], self.device)
                        if final_token_pos is not None:
                            break
                        else:
                            print(f'Level {self.tokenizer.decode(level_id)} not found in generated_ids_trimmed')

                # Final check: If a valid position is found, decode and print the token
                if final_token_pos is not None:  # Ensure that a valid position is returned
                    final_token = generated_ids_trimmed[0][final_token_pos]
                    final_word = self.tokenizer.decode(final_token)
                    print(f"Found token at position {final_token_pos}: {final_word}")
                else:
                    print("No valid pattern found in the generated sequence.")
                    continue  # Skip to the next iteration if no valid pattern is found
                                
                # assert final_token in self.cal_ids_, f'final token is not in [" excellent", " good", " fair", " poor", " bad"], but is {final_word}'
                cal_logit = logit[final_token_pos][:, self.cal_ids_]
                output_logit = logit[final_token_pos][:, self.cal_ids_].squeeze().tolist()
                print(cal_logit)
                print(answer)
                cal_logits.append(cal_logit)
                logits_dict = defaultdict(
                    float, {level: val for level, val in zip(self.level, output_logit)}
                )

                # 构建结果字典
                output_logits.append(
                    {"filename": os.path.basename(path), "logits": logits_dict, "answer": answer}
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

    