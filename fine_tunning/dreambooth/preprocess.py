"""
preprocess.py
- raw_data 폴더 안의 jpg/json 쌍을 찾아
- 이미지 리사이즈 & 같은 이름 .txt 캡션 생성
"""

import os, glob, json
from PIL import Image
from tqdm import tqdm

RAW_DIR = "raw_data"
OUT_DIR = "dataset/style"
STYLE_TAGS = "storybook illustration, watercolor, pastel colors, thin ink lines, soft lighting"

def ensure_dir(path): os.makedirs(path, exist_ok=True)

def resize_and_copy(img_path, out_path, size=1024):
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    img.save(out_path, "PNG")

def main():
    ensure_dir(OUT_DIR)

    # jpg와 같은 이름의 json을 함께 탐색
    for img_path in tqdm(glob.glob(os.path.join(RAW_DIR, "*.jpg"))):
        base = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(RAW_DIR, base + ".json")
        out_img = os.path.join(OUT_DIR, base + ".png")
        out_txt = os.path.join(OUT_DIR, base + ".txt")

        # 이미지 저장
        resize_and_copy(img_path, out_img)

        # JSON 캡션 추출
        scene_caption = ""
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                meta = json.load(f)
                scene_caption = meta.get("imageCaption", "").strip()

        # 스타일 태그 + 장면 설명 결합
        caption = f"{STYLE_TAGS}, {scene_caption}"
        with open(out_txt, "w", encoding="utf-8") as w:
            w.write(caption)

if __name__ == "__main__":
    main()
