import argparse
import json
import random

def make_image_prompts(outline):
    prompts = []
    for page in outline:
        seed = random.randint(10000, 99999)
        prompts.append({
            "page": page["page"],
            "image_prompt": f"{page['scene']} | CHAR:루루(노란 가방, 갈색 단발머리, 초록 운동화) | 따뜻한 파스텔 수채화",
            "negative_prompt": "텍스트 삽입, 공포스러운 요소, 손가락 왜곡",
            "seed": seed,
            "aspect_ratio": "4:3",
            "composition": "하단 우측 텍스트 공간 20%",
            "style": "밝고 따뜻한 동화풍"
        })
    return prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        outline = json.load(f)

    prompts = make_image_prompts(outline)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)

    print(f"[완료] 삽화 프롬프트 저장: {args.output}")
