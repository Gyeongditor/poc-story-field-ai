"""
main.py
- 커맨드라인 인자로 한국어 문장을 받고
- 영어 번역을 자동 추가하여
- 학습한 LoRA 동화풍 모델로 이미지 생성
"""

import os
import argparse
import torch
from diffusers import DiffusionPipeline
from googletrans import Translator

def generate_story_image(korean_sentence: str,
                         lora_path="outputs/style_lora",
                         out_file="samples/story_image.png"):

    # 1) 한국어 → 영어 번역
    translator = Translator()
    eng_trans = translator.translate(korean_sentence, src="ko", dest="en").text

    # 2) 모델 로드
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16
    ).to("cuda")

    # 3) LoRA 동화풍 가중치 로드
    pipe.load_lora_weights(lora_path)

    # 4) 프롬프트 결합 (한글 + 영어 + 스타일)
    style_tags = "storybook illustration, watercolor, pastel colors, thin ink lines, soft lighting"
    full_prompt = f"{korean_sentence}, {eng_trans}, {style_tags}"

    # 5) 이미지 생성
    image = pipe(
        full_prompt,
        num_inference_steps=32,
        guidance_scale=7,
        width=1024,
        height=768
    ).images[0]

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    image.save(out_file)
    print(f"이미지 저장 완료: {out_file}")
    print(f"번역된 프롬프트: {eng_trans}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="동화풍 이미지 생성")
    parser.add_argument(
        "-sentence",
        required=True,
        help="동화풍으로 만들 문장 (한국어 가능)"
    )
    parser.add_argument(
        "--lora_path",
        default="outputs/style_lora",
        help="학습된 LoRA 가중치 경로"
    )
    parser.add_argument(
        "--out_file",
        default="samples/story_image.png",
        help="출력 이미지 파일 경로"
    )
    args = parser.parse_args()

    generate_story_image(
        korean_sentence=args.sentence,
        lora_path=args.lora_path,
        out_file=args.out_file
    )
