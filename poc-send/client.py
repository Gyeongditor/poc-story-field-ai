import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple

import requests
import re


def discover_default_story_file() -> Path:
    """Try to find a reasonable default story file path within the repository."""
    candidate_paths = [
        Path("integration_test/result/gen_story/story.txt"),
        Path("STORY/StoryLine/generated_story.txt"),
        Path("STORY/GPT_API/travel_test_whisper.txt"),
    ]
    for path in candidate_paths:
        if path.exists() and path.is_file():
            return path
    # Fallback to a placeholder in poc-send dir
    fallback = Path(__file__).resolve().parent / "sample_story.txt"
    if not fallback.exists():
        fallback.write_text("이것은 샘플 스토리입니다.", encoding="utf-8")
    return fallback


def collect_image_files(images_dir: Path) -> List[Path]:
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    collected: List[Path] = []
    for pattern in patterns:
        collected.extend([Path(p) for p in glob.glob(str(images_dir / pattern))])
    # Stable ordering
    collected.sort()
    return collected


def build_multipart_payload(image_paths: List[Path], story_path: Path) -> List[Tuple[str, Tuple[str, bytes, str]]]:
    files: List[Tuple[str, Tuple[str, bytes, str]]] = []
    for image_path in image_paths:
        mime = "image/png"
        if image_path.suffix.lower() in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif image_path.suffix.lower() == ".webp":
            mime = "image/webp"
        files.append(
            (
                "files",
                (
                    image_path.name,
                    image_path.read_bytes(),
                    mime,
                ),
            )
        )

    # Story part
    files.append(
        (
            "story",
            (
                story_path.name,
                story_path.read_bytes(),
                "text/plain; charset=utf-8",
            ),
        )
    )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="POC: Send images + story via multipart/form-data")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Legacy simple upload
    p_simple = subparsers.add_parser("simple", help="Send simple images + story file")
    p_simple.add_argument("--endpoint", default="http://127.0.0.1:8000/upload", help="Receiver endpoint URL")
    p_simple.add_argument("--images-dir", default=str(Path("IMAGE/generated_images")), help="Directory containing images to send")
    p_simple.add_argument("--story-file", default=None, help="Path to a story text file. If omitted, a default will be discovered.")
    p_simple.add_argument("--title", default="Sample Story", help="Title metadata")
    p_simple.add_argument("--author", default=os.getlogin() if hasattr(os, "getlogin") else "unknown", help="Author metadata")
    p_simple.add_argument("--meta", default="{}", help="Optional metadata string (e.g., JSON)")

    # Story upload with uuid/title/thumbnail/page_texts/page_images
    p_story = subparsers.add_parser("story", help="Send structured story payload")
    p_story.add_argument("--endpoint", default="http://127.0.0.1:8000/upload-story", help="Receiver endpoint URL")
    p_story.add_argument("--uuid", required=True, help="Story UUID")
    p_story.add_argument("--title", required=True, help="Story title")
    p_story.add_argument("--thumbnail", required=True, help="Path to thumbnail image")
    p_story.add_argument("--page-texts", required=True, help="Path to a UTF-8 text file where each line is a page text")
    p_story.add_argument("--page-images-dir", required=True, help="Directory containing per-page images (sorted by filename)")
    parser.add_argument(
        "--images-dir",
        default=str(Path("IMAGE/generated_images")),
        help="Directory containing images to send",
    )
    parser.add_argument(
        "--story-file",
        default=None,
        help="Path to a story text file. If omitted, a default will be discovered.",
    )
    parser.add_argument("--title", default="Sample Story", help="Title metadata")
    parser.add_argument("--author", default=os.getlogin() if hasattr(os, "getlogin") else "unknown", help="Author metadata")
    parser.add_argument("--meta", default="{}", help="Optional metadata string (e.g., JSON)")

    args = parser.parse_args()

    if args.command == "story":
        endpoint = args.endpoint
        uuid = args.uuid
        title = args.title
        thumbnail_path = Path(args.thumbnail)
        page_texts_file = Path(args["page_texts"]) if isinstance(args, dict) else Path(getattr(args, "page_texts"))
        page_images_dir = Path(args["page_images_dir"]) if isinstance(args, dict) else Path(getattr(args, "page_images_dir"))

        if not thumbnail_path.exists():
            raise SystemExit(f"썸네일이 없습니다: {thumbnail_path}")
        if not page_texts_file.exists():
            raise SystemExit(f"페이지 텍스트 파일이 없습니다: {page_texts_file}")
        if not page_images_dir.exists() or not page_images_dir.is_dir():
            raise SystemExit(f"페이지 이미지 디렉터리가 없습니다: {page_images_dir}")

        # Read page texts using [PAGE] delimiters
        content = page_texts_file.read_text(encoding="utf-8")
        # Split on lines that contain only [PAGE] (allow trailing spaces)
        segments = re.split(r"(?m)^\[PAGE\]\s*\n?", content)
        page_texts = []
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            page_texts.append(seg)
        image_paths = collect_image_files(page_images_dir)

        files: List[Tuple[str, Tuple[str, bytes, str]]] = []
        # thumbnail
        thumb_mime = "image/png"
        if thumbnail_path.suffix.lower() in {".jpg", ".jpeg"}:
            thumb_mime = "image/jpeg"
        elif thumbnail_path.suffix.lower() == ".webp":
            thumb_mime = "image/webp"
        files.append(("thumbnail", (thumbnail_path.name, thumbnail_path.read_bytes(), thumb_mime)))

        # page images
        for image_path in image_paths:
            mime = "image/png"
            if image_path.suffix.lower() in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            elif image_path.suffix.lower() == ".webp":
                mime = "image/webp"
            files.append(("page_images", (image_path.name, image_path.read_bytes(), mime)))

        # data fields: uuid, title, page_texts (repeated)
        data = [("uuid", uuid), ("title", title)]
        for text in page_texts:
            data.append(("page_texts", text))

        print(f"POST {endpoint}")
        print(f"- uuid: {uuid}")
        print(f"- title: {title}")
        print(f"- page_texts: {len(page_texts)}개")
        print(f"- page_images: {len(image_paths)}개")

        response = requests.post(endpoint, files=files, data=data, timeout=120)
        response.raise_for_status()
        print("서버 응답:")
        print(response.text)
        return

    # default: simple
    endpoint = getattr(args, "endpoint", "http://127.0.0.1:8000/upload")
    images_dir = Path(getattr(args, "images_dir", "IMAGE/generated_images"))
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"이미지 디렉터리가 없습니다: {images_dir}")

    image_paths = collect_image_files(images_dir)
    if not image_paths:
        raise SystemExit(f"이미지 파일을 찾지 못했습니다: {images_dir}")

    story_file = getattr(args, "story_file", None)
    story_path = Path(story_file) if story_file else discover_default_story_file()
    if not story_path.exists():
        raise SystemExit(f"스토리 파일을 찾지 못했습니다: {story_path}")

    files = build_multipart_payload(image_paths, story_path)
    data = {
        "title": getattr(args, "title", "Sample Story"),
        "author": getattr(args, "author", os.getlogin() if hasattr(os, "getlogin") else "unknown"),
        "meta": getattr(args, "meta", "{}"),
    }

    print(f"POST {endpoint}")
    print(f"- images: {len(image_paths)}개")
    print(f"- story: {story_path}")

    response = requests.post(endpoint, files=files, data=data, timeout=60)
    response.raise_for_status()
    print("서버 응답:")
    print(response.text)


if __name__ == "__main__":
    main()


