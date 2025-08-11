from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import os
import requests


app = FastAPI(title="POC Multipart Receiver", version="0.1.0")

# Spring backend base URL, e.g. http://127.0.0.1:8080
SPRING_BASE_URL = os.getenv("SPRING_BASE_URL", "http://127.0.0.1:8080").rstrip("/")


def ensure_directory_exists(target_dir: Path) -> None:
    """Create directory recursively if it does not exist."""
    target_dir.mkdir(parents=True, exist_ok=True)


@app.post("/upload")
async def upload(
    files: Optional[List[UploadFile]] = File(default=None, description="Multiple image files"),
    story: Optional[UploadFile] = File(default=None, description="Story text file"),
    title: Optional[str] = Form(default=None),
    author: Optional[str] = Form(default=None),
    meta: Optional[str] = Form(default=None, description="Optional JSON string or arbitrary text metadata"),
):
    """Receive images and a story file via multipart/form-data and save them under timestamped folder.

    - Saves to: poc-send/uploads/{YYYYmmdd-HHMMSS}/
      - images/...
      - story.txt (if provided)
      - meta.txt (if provided)
      - info.json summary response is returned to the client
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = Path(__file__).resolve().parent / "uploads" / timestamp
    images_dir = base_dir / "images"
    ensure_directory_exists(images_dir)

    saved_images: List[str] = []

    # Save images if provided
    if files:
        for index, upload_file in enumerate(files):
            original_name = upload_file.filename or f"image_{index}"
            extension = os.path.splitext(original_name)[1] or ".bin"
            safe_name = f"{index:03d}_{Path(original_name).name}"
            target_path = images_dir / safe_name
            content = await upload_file.read()
            target_path.write_bytes(content)
            saved_images.append(str(target_path))

    # Save story if provided
    saved_story: Optional[str] = None
    if story is not None:
        story_name = story.filename or "story.txt"
        target_story_path = base_dir / story_name
        content = await story.read()
        target_story_path.write_bytes(content)
        saved_story = str(target_story_path)

    # Save meta if provided
    saved_meta: Optional[str] = None
    if meta:
        meta_path = base_dir / "meta.txt"
        meta_path.write_text(meta, encoding="utf-8")
        saved_meta = str(meta_path)

    # Save submission info
    info = {
        "title": title,
        "author": author,
        "meta_saved_path": saved_meta,
        "story_saved_path": saved_story,
        "image_saved_paths": saved_images,
        "upload_dir": str(base_dir),
        "num_images": len(saved_images),
    }

    return JSONResponse(content=info)


@app.post("/upload-story")
async def upload_story(
    uuid: str = Form(..., description="Story UUID"),
    title: str = Form(..., description="Story title"),
    thumbnail: UploadFile = File(..., description="Thumbnail image"),
    page_texts: List[str] = Form(default=[], description="Repeated field for each page text"),
    page_images: List[UploadFile] = File(default=[], description="Repeated file field for each page image"),
):
    """Receive a story payload consisting of uuid, title, thumbnail, per-page texts and per-page images.

    Expected multipart fields:
    - uuid: str (Form)
    - title: str (Form)
    - thumbnail: UploadFile (File)
    - page_texts: List[str] (Form, repeated field name)
    - page_images: List[UploadFile] (File, repeated field name)
    """
    # Base directory: prefer uuid for determinism
    base_dir = Path(__file__).resolve().parent / "uploads" / uuid
    pages_dir = base_dir / "pages"
    ensure_directory_exists(pages_dir)

    # Save thumbnail
    thumb_name = thumbnail.filename or "thumbnail"
    thumb_ext = os.path.splitext(thumb_name)[1] or ".bin"
    thumb_path = base_dir / f"thumbnail{thumb_ext}"
    thumb_bytes = await thumbnail.read()
    thumb_path.write_bytes(thumb_bytes)

    # Pair page texts and images by index
    num_pairs = min(len(page_texts), len(page_images))
    saved_pages = []
    for index in range(num_pairs):
        # Text
        text_content = page_texts[index]
        text_path = pages_dir / f"{index + 1:03d}.txt"
        text_path.write_text(text_content, encoding="utf-8")

        # Image
        img_upload = page_images[index]
        img_name = img_upload.filename or f"page_{index + 1}"
        img_ext = os.path.splitext(img_name)[1] or ".bin"
        img_path = pages_dir / f"{index + 1:03d}{img_ext}"
        img_bytes = await img_upload.read()
        img_path.write_bytes(img_bytes)

        saved_pages.append({
            "index": index + 1,
            "text_path": str(text_path),
            "image_path": str(img_path),
        })

    info = {
        "uuid": uuid,
        "title": title,
        "thumbnail_path": str(thumb_path),
        "num_page_texts_received": len(page_texts),
        "num_page_images_received": len(page_images),
        "num_pages_saved": len(saved_pages),
        "pages": saved_pages,
        "upload_dir": str(base_dir),
        "note": "Saved pages are paired by index; extras (if any) are ignored in this POC.",
    }

    return JSONResponse(content=info)


@app.post("/forward-story")
async def forward_story(
    uuid: str = Form(..., description="Story UUID"),
    title: str = Form(..., description="Story title"),
    thumbnail: UploadFile = File(..., description="Thumbnail image"),
    page_texts: List[str] = Form(default=[], description="Repeated field for each page text"),
    page_images: List[UploadFile] = File(default=[], description="Repeated file field for each page image"),
):
    """Forward the received story payload to Spring backend /upload as multipart/form-data.

    Environment:
    - SPRING_BASE_URL (default: http://127.0.0.1:8080)
    """
    spring_endpoint = f"{SPRING_BASE_URL}/upload"

    # Build multipart
    # data: uuid, title, page_texts (repeated)
    data: List[tuple[str, str]] = [("uuid", uuid), ("title", title)]
    for txt in page_texts:
        data.append(("page_texts", txt))

    # files: thumbnail (single), page_images (repeated)
    def _guess_mime(name: str, fallback: str = "application/octet-stream") -> str:
        lower = name.lower()
        if lower.endswith((".png",)):
            return "image/png"
        if lower.endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        if lower.endswith((".webp",)):
            return "image/webp"
        if lower.endswith((".txt",)):
            return "text/plain; charset=utf-8"
        return fallback

    files: List[tuple[str, tuple[str, bytes, str]]] = []
    thumb_name = thumbnail.filename or "thumbnail"
    thumb_bytes = await thumbnail.read()
    files.append(("thumbnail", (thumb_name, thumb_bytes, thumbnail.content_type or _guess_mime(thumb_name))))

    for img in page_images:
        img_name = img.filename or "page_image"
        img_bytes = await img.read()
        files.append(("page_images", (img_name, img_bytes, img.content_type or _guess_mime(img_name))))

    try:
        resp = requests.post(spring_endpoint, data=data, files=files, timeout=120)
        content_type = resp.headers.get("content-type", "")
        try:
            body = resp.json() if "application/json" in content_type else resp.text
        except Exception:
            body = resp.text
        return JSONResponse(
            status_code=resp.status_code,
            content={
                "forwarded_to": spring_endpoint,
                "spring_status_code": resp.status_code,
                "spring_body": body,
            },
        )
    except requests.RequestException as exc:
        return JSONResponse(
            status_code=502,
            content={
                "forwarded_to": spring_endpoint,
                "error": "Failed to reach Spring backend",
                "detail": str(exc),
            },
        )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


