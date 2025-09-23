import argparse
import json
import torch
from diffusers import StableDiffusionPipeline

def generate_images(prompts, output_dir="images", model_id="runwayml/stable-diffusion-v1-5"):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    images = []
    for page in prompts:
        print(f"[생성 중] Page {page['page']}...")
        image = pipe(
            prompt=page["image_prompt"],
            negative_prompt=page["negative_prompt"],
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=torch.manual_seed(page["seed"])
        ).images[0]

        out_path = f"{output_dir}/page_{page['page']}.png"
        image.save(out_path)
        images.append({"page": page["page"], "file": out_path})

    return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="img_prompts.json")
    parser.add_argument("--output_dir", default="images", help="저장할 폴더명")
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5", help="HuggingFace 모델 ID")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    results = generate_images(prompts, output_dir=args.output_dir, model_id=args.model_id)

    print("[완료] 이미지 생성 완료")
    for r in results:
        print(f"Page {r['page']} → {r['file']}")
