import argparse
import base64
import json
import os

import weave
from openai import OpenAI
from PIL import Image
import random
from data_utils import load_json, encode_img, conver_meta_data_to_gpt
from collections import Counter


OPENAI_API_KEY = "sk-tE7K8vJ9Dla5zDMx87F9EeB7372340C68067179938991e54"
OPENAI_API_BASE = "https://api.gpt.ge/v1"
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

parser = argparse.ArgumentParser(
    description="To Prompt GPT-4 for Image Quality Assessment"
)
parser.add_argument("--meta_file", type=str, default='data/meta_json/benchmark/test/input_dist_info_v1.json')
parser.add_argument("--desp_file", type=str, default='data/meta_json/description.json')
parser.add_argument("--assess_file", type=str, default='data/meta_json/benchmark/test/test_assess_v1.json')
parser.add_argument("--image_folder", type=str, default='../datasets/images/gvlmiqa_bench/')
example_image = "../datasets/images/gvlmiqa_bench/LIVEfb_AVA__69656.jpg"
example_meta_input = "Assess the overall quality of this image and each distortion based on the following image information and distortion information:\n[Image Infomation]\nMos: 2.8, Level: fair, Description: The image depicts a young boy riding a mechanical horse, typically found in playgrounds or amusement areas. The boy is wearing a striped shirt and appears to be enjoying the ride. In the foreground, there is a silhouette of another child watching. The background shows an outdoor setting with trees and a blurred landscape.\n[Distortion Infomation]\n<bbox 1> Distortion: Out of focus blur, Position: The saddle of the horse and the boy's legs, Severity: moderate, Visual Manifestation: In this region, the out of focus blur causes the edges of the saddle and the boy's legs to appear soft and lacking in detail. The texture of the saddle is not discernible, and the legs of the boy blend into the saddle, losing definition., Perception Impact: The out of focus blur significantly diminishes the clarity and detail of the horse's saddle and the boy's legs, making it difficult to discern texture and sharpness. The overall composition is affected as the blur reduces the visual separation between the boy and the horse., Region Quality: The quality is bad., Region Importance: This region is minor to the overall quality.\n<bbox 2> Distortion: Out of focus blur, Position: The silhouette of the person in the foreground, Severity: severe, Visual Manifestation: The silhouette appears with soft, indistinct edges, blending into the background. The lack of sharpness and definition in this area causes the silhouette to lose its visual impact and presence in the image., Perception Impact: The out of focus blur severely affects the silhouette, causing it to blend into the background and lose its distinct shape and texture. The lack of visibility impacts the overall composition and focus of the image., Region Quality: The quality is fair., Region Importance: This region is important to the overall quality.\n<bbox 3> Distortion: Out of focus blur, Position: The boy's striped shirt and the front part of the horse, Severity: moderate, Visual Manifestation: The stripes on the boy's shirt are indistinct, appearing as blurred lines with little contrast. The front part of the horse lacks definition, with soft edges and a loss of textural detail., Perception Impact: The out of focus blur reduces the visibility of the boy's striped shirt and the front part of the horse, making it hard to distinguish the pattern and details. This affects the perception of depth and detail in the central part of the image., Region Quality: The quality is poor., Region Importance: This region is minor to the overall quality.\n<bbox 4> Distortion: Edge aliasing effect, Position: The tree branches in the background, Severity: minor, Visual Manifestation: The edges of the tree branches appear jagged and pixelated, rather than smooth and natural. This creates a distracting effect that detracts from the natural appearance of the landscape., Perception Impact: The edge aliasing effect introduces visual noise that affects the clarity and sharpness of the tree branches, making them appear jagged and unnatural. This affects the perception of the natural landscape in the background., Region Quality: The quality is poor., Region Importance: This region is normal to the overall quality.\n<bbox 5> Distortion: Underexposure, Position: The right side of the image, including parts of the horse and background, Severity: severe, Visual Manifestation: The affected area is dark, with little to no visible detail. The underexposure causes a significant loss of texture and clarity, making the objects in this region appear as dark, indistinct shapes., Perception Impact: The underexposure causes the right side of the image, including parts of the horse and the background, to appear dark and lacking in detail, which reduces visibility and impacts the overall lighting balance of the image., Region Quality: The quality is poor., Region Importance: This region is unimportant to the overall quality.\n<bbox 6> Distortion: Underexposure, Position: The silhouette of the person in the foreground, Severity: severe, Visual Manifestation: The silhouette is poorly defined, appearing as a dark shape with no visible features. The underexposure causes a loss of detail and texture, making the silhouette indistinct and blending it into the background., Perception Impact: The underexposure in this region makes the silhouette blend into the background, reducing its visibility and detail. The lack of contrast affects the silhouette's distinctiveness and the overall composition of the image., Region Quality: The quality is poor., Region Importance: This region is normal to the overall quality.\n"

overall_quality_assessment = "The image depicts a young boy riding a mechanical horse in an outdoor setting, with another child silhouetted in the foreground.\nThere are three out of foucs blurs in the image. <bbox 1>Moderate out of focus blur</bbox 1> causes the edges of the saddle and the boy's legs to appear soft and lacking in detail. The texture of the saddle is not discernible, and the legs of the boy blend into the saddle, losing definition.<bbox 2>Severe out of focus blur</bbox 2> made the silhouette appear with soft, indistinct edges, blending into the background. The lack of sharpness and definition in this area causes the silhouette to lose its visual impact and presence in the image. The boy's striped shirt and the front part of the horse are also affected by <bbox 3>moderate out of foucs blur</bbox 3>.The stripes on the boy's shirt are indistinct, appearing as blurred lines with little contrast. The front part of the horse lacks definition, with soft edges and a loss of textural detail.\nThere is one <bbox 4>minor edge aliasing effect</bbox 4> affecting tree branches in the background.The edges of the tree branches appear jagged and pixelated, rather than smooth and natural. This creates a distracting effect that detracts from the natural appearance of the landscape.\nThere are two underexposures in the image. <bbox 5>Severe underexposure</bbox 5> causes the right side of the image, including parts of the horse and the background, to appear dark and lacking in detail. The silhouette of the person in the foreground also shows <bbox 6>severe underexposure</bbox 6>. The silhouette is poorly defined, appearing as a dark shape with no visible features. The underexposure causes a loss of detail and texture, making it indistinct and blending it into the background.\nThe <bbox 2>severe out of focus blur</bbox 2> and <bbox 6>severe underexposure</bbox 6> in the foreground have a significant impact on the overall quality. The foreground is important to the overall quality, but it appears blurry and lacks contrast, weakening the sense of visual focus and composition.The overall contrast is weak, with object outlines appearing soft and indistinct. However, no significant noise is observed, and the image maintains a monochromatic appearance without any noticeable color distortions.\nThus, the quality of the image is fair."
distortion_assessment = [
    {
        "name": "Out of focus blur",
        "assessment": "There are three out of foucs blurs in the image.\n<bbox 1>Moderate out of focus blur</bbox 1> affects the horse's saddle and the boy's legs, resulting in a loss of clarity and detail. The saddle's texture becomes indiscernible, appearing flat and devoid of any material characteristics. Similarly, the boy's legs lose definition and blend into the saddle, making it difficult to distinguish the two elements. The softening of edges reduces the visual separation between the boy and the horse, which diminishes the sense of interaction and depth in the scene. The affected area quality is bad due to its significant detail loss.However, its importance to the overall composition is minor. While the blur does impact the visual harmony of the scene, it has a relatively limited effect on the image's overall perception, given its secondary role in the composition.\n<bbox 2>Severe out of focus blur</bbox 2> affects the foreground silhouette.The edges of the silhouette are soft and indistinct, causing it to blend into the background entirely. This lack of sharpness significantly reduces the silhouette's visual presence and eliminates its potential as a strong focal point. The inability to clearly define this area weakens the image's composition and disrupts its visual focus. The affected area quality is fair, but it is deemed highly important to the image's overall quality due to its position in the foreground. The severe blur in this critical area not only detracts from the foreground's role as a compositional anchor but also diminishes the sense of depth and separation between foreground and background. This has a profound negative effect on the viewer's ability to engage with the image.\n<bbox 3> Moderate Out of Focus Blur</bbox3>  affects the boy's striped shirt and the front part of the horse. The shirt's stripes, which could provide contrast and visual interest, are blurred into indistinct lines, reducing their prominence. Similarly, the front part of the horse suffers from soft edges and a lack of textural detail, making it appear less dynamic and lifelike. The blurred details in this central region reduce the sense of depth and textural richness in the image. The affected area quality is poor, but its importance to the overall composition is minor. While the loss of detail is noticeable, this area does not serve as a primary focus in the image, thus limiting the extent of its impact on the viewer's perception.\nThe out of focus blur across these three regions collectively undermines the image's clarity and depth. The severe blur has the most significant impact, as it affects the foreground—a critical compositional element—resulting in a loss of focus and weak spatial separation. The moderate blur also contributes to the degradation by softening key details in the central and supporting areas, though their impact is less pronounced due to their secondary importance. Together, these blurs create a flatter, less engaging image with diminished detail and definition, ultimately lowering the overall quality."
    },
    {
        "name": "Edge aliasing effect",
        "assessment": "There is one edge aliasing effect in the image.\n<bbox 4>Minor edge aliasing effect</bbox 4> affects the tree branches in the background. The edges of the branches appear jagged and pixelated, disrupting their natural smoothness. This introduces visual noise, making the tree branches look unnatural and detracting from the overall perception of the background landscape. The affected area quality is poor, as the unnatural appearance of the branches reduces the aesthetic appeal of the image. It is normal to overall image quality, as the background plays a supporting role in the composition. While the edge aliasing is noticeable, it does not critically affect the primary subject or the overall focus.\nThe edge aliasing effect in the tree branches adds a distracting visual element that impacts the perception of the natural landscape. However, given the minor severity and the secondary importance of the affected region, the overall impact on the image's quality is limited."
    },
    {
        "name": "Underexposure",
        "assessment": "There are two underexposures in the image.\n<bbox 5>Severe underexposure</bbox 5> affects the right side of the image, including parts of the horse and the background. The underexposure in this area causes significant darkening, resulting in a lack of detail and texture. The portions of the horse lose their visual definition, appearing as flat, shadowy shapes. The background becomes indistinct, with dark tones blending together and creating a lack of separation between elements. The affected area quality is poor, as the loss of detail and texture significantly diminishes the visual richness. It is moderate to overall image quality. While the underexposure reduces the balance and tonal depth of the image, it does not critically impact the primary focus.\n<bbox 6>Severe underexposure</bbox 6> impacts the silhouette of the person in the foreground. The silhouette is nearly featureless, appearing as a dark shape with no discernible texture or detail. The severe underexposure in this critical foreground region eliminates its potential to function as a strong visual anchor, causing it to blend into the dark background. The affected area quality is bad due to the complete loss of definition and texture. It is important to overall image quality, as the foreground is essential for establishing depth and focus. The lack of visibility and contrast disrupts the viewer's ability to engage with the image and perceive the intended composition.\nThe underexposure in these two regions has a significant impact on the image. The severe underexposure in the foreground is the most detrimental, as it eliminates the possibility of a strong focal point and weakens the overall composition. Meanwhile, the underexposure in the right side of the image reduces balance and diminishes the richness of supporting details. Together, these issues create a dark, visually flat image with reduced definition and contrast, ultimately lowering the overall quality."        
    }
]
example_gpt_output = {"overall quality assessment": overall_quality_assessment, "single distortion assessment": distortion_assessment}
example_gpt_output_str = json.dumps(example_gpt_output)
print(type(example_gpt_output))
print(type(example_gpt_output_str))
# print(example_meta_input)
# print(example_gpt_output)

# Weave will track the inputs, outputs and code of this function
@weave.op()
def gpt4o(img_path, query, system_prompt, with_dist):
    if with_dist:
        resp_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "analysis_response_schema",
                "strict": True,
                "schema": {
                "type": "object",
                "properties": {
                    "overall quality assessment": {
                        "type": "string",
                        "description": "The overall quality assessment of the image."
                    },
                    "single distortion assessment": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The name of the distortion."
                                },
                                "assessment": {
                                    "type": "string",
                                    "description": "The assessment of the distortion."
                                }
                            },
                            "required": ["name", "assessment"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["overall quality assessment", "single distortion assessment"],
                "additionalProperties": False
            }
        }
        }
        resp = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": example_meta_input},
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": example_gpt_output_str},
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                    ],
                }
            ],
            temperature=0.5,
            response_format=resp_format
        )
    else:
        mime_type, img_base64 = encode_img(img_path)
        print("Encoded image data length:", len(img_base64))  # 添加调试信息，打印前30个字符        
        resp = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{img_base64}"
                                },
                        },
                    ],
                }
            ],
            temperature=0.5
        )
    print(resp.usage)
    content = resp.choices[0].message.content
    return content

def check(meta_data, assess_data):
    # check if all images have been generated
    img_names = [item["image"] for item in meta_data]
    assess_names = [item["image"] for item in assess_data]
    if set(img_names) == set(assess_names):
        print("All images have been generated.")
    else:    
        len_diff = len(img_names) - len(assess_names)
        if len_diff > 0:
            print(f"There are {len_diff} images not generated.")
def check_duplicate(meta_data):
    # check if there are duplicate images
    file_names = [item["image"] for item in meta_data]
    file_name_counts = Counter(file_names)
    duplicates = {name: count for name, count in file_name_counts.items() if count > 1}
    if duplicates:
        print("Duplicate images found:")
        for name, count in duplicates.items():
            print(f"{name}: {count} times")
    else:
        print("No duplicate images found.")

if __name__ == "__main__":
    weave.init("image quality assessment")
    args = parser.parse_args()
    idx_meta_start = 0
    idx_meta_end = -1

    meta_file = args.meta_file
    desp_file = args.desp_file
    assess_file = args.assess_file
    image_folder = args.image_folder

    meta_data = load_json(meta_file)
    if os.path.exists(desp_file):
        desp_data = load_json(desp_file)
    else:
        print("Please generate description first")
    if os.path.exists(assess_file):
        assess_data = load_json(assess_file)
    else:
        assess_data = []

    check_flag = check(meta_data, assess_data)
    check_duplicate(meta_data)
    dist_paths_error = []
    
    for idx_meta, meta_item in enumerate(meta_data[idx_meta_start:]):
        img_name = meta_item["image"]
        if img_name in [item["image"] for item in assess_data]:
            # print(f"{img_name} has been generated, skip.")
            continue
        print("=" * 100)
        print(idx_meta + idx_meta_start)
        print(img_name)
        img_path = os.path.join(image_folder, img_name)
        desp_item = next(
            (item for item in desp_data if item["filename"] == img_name), None
        )
        if desp_item:
            description_parts = desp_item["gpt4v_description"].split("\n\n")
            description_parts = (
                description_parts
                if len(description_parts) > 1
                else desp_item["gpt4v_description"].split("\n")
            )
            description = (
                description_parts[1]
                .replace("**Answer:** ", "")
                .replace("**Answer:**", "")
                .replace("Answer: ", "")
                .strip()  # This removes leading or trailing spaces
                if len(description_parts) > 1
                else description_parts
            )
        else:
            print(f"{img_name} has no description, please generate description first.")
            continue 
        if isinstance(meta_item['distortions'], list): 
            with_dist = True
            try:
                query = conver_meta_data_to_gpt(meta_item, description[0])
            except Exception as e:
                print("Description:", description)
                print(f"Error occurred while generating query for {img_name}: {e}")
                query = conver_meta_data_to_gpt(meta_item, None)
            # print(query)
            system_prompt =(
                " You are an expert in image quality assessment. Your task is to assess the overall quality of an image and each distortion based on image information and distortion information provided to you."
                + " The image information includes a Mean Opinion Score (MOS) out of 5, the image quality level(one of bad, poor, fair, good, excellent) and a description of the image."
                + " Use the image information as the basis for your assessment, but do not mention or imply terms like 'rating', 'MOS', or 'quality level'."
                + " The distortion information includes the distortion id, distortion name, position, severity, region quality, region importance, visual manifestation and perception impact."
                + " Do not refer to the distortion information as provided; instead, act as though your observations and evaluations are entirely derived from your own interpretation of the image "
                + " To assess the overall image quality, you should think step by step."
                + " **First step**, provide a brief description of the image content according to the description in one sentence."
                + " If the providing description is None, generate a one-sentence summary of the image content."
                + " If the description is unavailable, please provide a one-sentence summary explaining why the image content cannot be described."
                + " The description should be in a declarative form, without using the first person."
                + " **Second step**, briefly analyze each distortion(grouped by distortion name) according to the distortion information."
                + " Fully exploit the distortion information to explain how the distortion affect the visual appearance and perception of specific objects or regions in the image."
                + " Seperate each distortion explanation with a line break(\n)."
                + " **Third step**, identify the key distortions that have the most significant impact on the overall image quality and give your detailed reasoning."
                + " At the same time, summarize the low-level attributes of the image. Combine the analysis of key degradations and low-level attributes into a cohesive paragraph. Ensure this paragraph logically aligns with the provided information."
                + " **Final step**, end your answer with this sentence: Thus, the quality of the image is (the image quality level provided)."
                + " Seperate each step with a line break(\n)."
                + " To assess each distortion(grouped by distortion name), you also need to think step by step."
                + " **First step**, provide the number of occurrences of the distortion in the image."
                + " **Second step**, Analyze each distortion in detail: first, describe its position; then, assess its visual manifestation and perceptual impact; and finally, evaluate how the distortion affects the image by considering the region quality, region importance, and the image's low-level attributes."
                + " Seperate each distortion explanation with a line break(\n)."
                + " **Final step**, summarize the impact of this distortion on the overall quality of the image."
                + " Seperate each step with a line break(\n)."
                + " In both tasks, ensure every distortion entity is mentioned, and highlight the distortion entity(severity + distortion name, for example, severe motion blur) with corresponding <bbox id> in the resulting paragraph."
                + " Answer in a consistent style."
            )  
            assess_query = ( 
                "Assess the overall quality of this image and each distortion based on the following image information and distortion information:\n"
                + f"{query}"
            )
        else:
            with_dist = False
            level = meta_item['level']
            mos = meta_item['mos']
            system_prompt = (
                " You are an expert in image quality assessment. Your task is to assess the overall quality of an image and each distortion based on image information and distortion information provided to you."
                + " The image information includes a Mean Opinion Score (MOS) out of 5, the image quality level(one of bad, poor, fair, good, excellent) and a description of the image."
                + " The image contains no distortion."
                + " To assess the image quality, you should think step by step."
                + " First step, provide a brief description of the image content summarizing the description in one sentence. "
                + " Second step, analyze the overall image quality and visual perception by highlighting the strengths of the image quality and emphasizing the absence of image quality issues. "
                + " Start your analysis with 'There is no distortion in the image.'"
                + " In this part, use the provided MOS, quality level as the basis for your analysis."
                + " Focus solely on what you observe in the image. Do not mention or imply terms like 'rating', 'MOS', or 'quality level'."
                + " Instead, describe the image's visual aspects, such as visibility, detail discernment, clarity, brightness, lighting, composition, and texture etc."
                + " Your analysis should reflect the overall quality without referencing any scores or evaluation terms, as if you are unaware of them."
                + " Final step, end your answer with this sentence: Thus, the quality of the image is (the image quality level provided)."
                + " Seperate each step with a line break(\n)."
            )
            assess_query = (
                " Image information is as follows: "
                + f" The description of image content is: {description[0]}. "
                + f" The image is rated as {level} quality and has a Mean Opinion Score (MOS) of {mos} out of 5 based on evaluation standards. " 
                + " Your response must be concise, logically consistent, not exceeding 300 words."
            )

        try:
            print("Generating assessment...")
            content = gpt4o(img_path, assess_query, system_prompt, with_dist)
            print(content)
            if with_dist:
                resp_dict = json.loads(content)
                meta_item["assessment"] = resp_dict
            else:
                meta_item["assessment"] = content
            assess_data.append(meta_item)
            print(f"Assessment for {img_name} generated.")
            with open(assess_file, "w") as fw:
                json.dump(assess_data, fw, indent=4, ensure_ascii=False)
        except:
            import sys
            print("Error occurred while generating assessment for {}.".format(img_name))
            except_type, except_value, except_traceback = sys.exc_info()
            except_file = os.path.split(except_traceback.tb_frame.f_code.co_filename)[1]
            exc_dict = {
                "error type": except_type,
                "error info": except_value,
                "error file": except_file,
                "error line": except_traceback.tb_lineno,
            }
            print(exc_dict)
            dist_paths_error.append(img_name)
    fail_dir = os.path.dirname(assess_file)
    os.makedirs(fail_dir, exist_ok=True)
    fail_path = os.path.join(fail_dir, "res_fail.txt")
    with open(fail_path, "w") as fw:
        fw.write("\n".join(dist_paths_error))
