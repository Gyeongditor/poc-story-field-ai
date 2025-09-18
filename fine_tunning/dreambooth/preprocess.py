import os, glob, json, argparse
from PIL import Image, ImageOps
from PIL.Image import DecompressionBombError
from tqdm import tqdm

RAW_DIR = "raw_data"
OUT_DIR = "dataset/style"
STYLE_TAGS = "storybook illustration, watercolor, pastel colors, thin ink lines, soft lighting"

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def resize_and_copy(img_path, out_path, size=1024):
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    img.thumbnail((size, size), Image.LANCZOS)
    img.save(out_path, "PNG")

def find_images(raw_dir, extensions):
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(raw_dir, f"*.{ext}")))
    return sorted(paths)

def main():
    parser = argparse.ArgumentParser(description="Style LoRA 전처리: 이미지 리사이즈 및 캡션 생성")
    parser.add_argument("--raw_dir", default=RAW_DIR, help="원천 이미지/메타 경로")
    parser.add_argument("--out_dir", default=OUT_DIR, help="전처리 산출 경로 (png/txt)")
    parser.add_argument("--style_tags", default=STYLE_TAGS, help="스타일 캡션 태그")
    parser.add_argument("--size", type=int, default=1024, help="긴 변 기준 최대 해상도")
    parser.add_argument("--exts", default="jpg,jpeg,png", help="허용 확장자(쉼표 구분)")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]

    skipped_huge = []
    skipped_existing = []

    for img_path in tqdm(find_images(args.raw_dir, exts)):
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_img = os.path.join(args.out_dir, base + ".png")
        out_txt = os.path.join(args.out_dir, base + ".txt")
        json_path = os.path.join(args.raw_dir, base + ".json")

        # --- 이미 처리된 파일 스킵 ---
        if os.path.exists(out_img) and os.path.exists(out_txt):
            skipped_existing.append(img_path)
            continue

        # --- 이미지 리사이즈 ---
        try:
            resize_and_copy(img_path, out_img, size=args.size)
        except DecompressionBombError:
            print(f"[SKIP-HUGE] {img_path}")
            skipped_huge.append(img_path)
            continue
        except Exception as e:
            print(f"[SKIP-ERR] {img_path} - {e}")
            skipped_huge.append(img_path)
            continue

        # --- JSON 캡션 ---
        scene_caption = ""
        if os.path.exists(json_path):
            try:
                with open(json_path, encoding="utf-8") as f:
                    meta = json.load(f)
                    scene_caption = meta.get("imageCaption", "").strip()
            except Exception as e:
                print(f"[WARN] JSON 읽기 실패: {json_path} - {e}")

        caption = f"{args.style_tags}, {scene_caption}" if scene_caption else args.style_tags
        with open(out_txt, "w", encoding="utf-8") as w:
            w.write(caption)

    # --- 요약 로그 ---
    if skipped_existing:
        print("\n=== 이미 처리되어 스킵된 이미지 ===")
        for p in skipped_existing:
            print(p)
    if skipped_huge:
        print("\n=== 초대형/에러로 스킵된 이미지 ===")
        for p in skipped_huge:
            print(p)

if __name__ == "__main__":
    main()
