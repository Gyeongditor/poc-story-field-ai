import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import glob
import requests


def guess_mime_type(file_path: Path) -> str:
    """Return a best-effort MIME type based on file extension.

    Defaults to application/octet-stream when unknown.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "application/octet-stream"


def parse_kv_pairs(pairs: list[str]) -> Dict[str, str]:
    """Parse CLI --data key=value pairs into a dict.

    Ignores malformed entries safely.
    """
    result: Dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            result[key] = value
    return result


def build_file_part(field_name: str, file_path: Path, override_mime: str | None = None) -> Tuple[str, Tuple[str, bytes, str]]:
    """Build a single (field, (filename, bytes, mime)) tuple for requests."""
    mime = override_mime or guess_mime_type(file_path)
    return (
        field_name,
        (
            file_path.name,
            file_path.read_bytes(),
            mime,
        ),
    )


def collect_image_files(directory: Path, patterns: List[str]) -> List[Path]:
    collected: List[Path] = []
    for pattern in patterns:
        collected.extend([Path(p) for p in glob.glob(str(directory / pattern))])
    collected = [p for p in collected if p.is_file()]
    collected.sort()
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send image(s) to an endpoint via multipart/form-data",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/upload",
        help="Target endpoint URL (default: http://127.0.0.1:8000/upload)",
    )
    parser.add_argument(
        "--image",
        required=False,
        help="Path to an image file (png/jpg/jpeg/webp/gif)",
    )
    parser.add_argument(
        "--dir",
        required=False,
        help="Directory containing images to send",
    )
    parser.add_argument(
        "--pattern",
        default="*.png,*.jpg,*.jpeg,*.webp,*.gif",
        help="Comma-separated glob patterns for --dir (default: *.png,*.jpg,*.jpeg,*.webp,*.gif)",
    )
    parser.add_argument(
        "--field-name",
        default="files",
        help="Multipart field name for the file(s) (default: files)",
    )
    parser.add_argument(
        "--mime-type",
        default=None,
        help="Override MIME type (e.g., image/png). If omitted, inferred from extension.",
    )
    parser.add_argument(
        "--data",
        action="append",
        default=[],
        help="Additional form fields as key=value (can be repeated)",
    )
    parser.add_argument(
        "--auth-bearer",
        default=None,
        help="Authorization bearer token. If set, sends 'Authorization: Bearer <TOKEN>'",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Additional headers as Key:Value (can be repeated)",
    )
    parser.add_argument(
        "--bulk",
        action="store_true",
        help="When using --dir, send all images in one request as repeated file fields",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)",
    )

    args = parser.parse_args()

    endpoint: str = args.endpoint
    field_name: str = args.field_name
    override_mime: str | None = args.mime_type
    extra_data = parse_kv_pairs(args.data)

    # Build headers
    headers: Dict[str, str] = {}
    if args.auth_bearer:
        headers["Authorization"] = f"Bearer {args.auth_bearer}"
    # parse --header Key:Value pairs
    for raw in args.header or []:
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key:
            headers[key] = value

    # Determine input mode: single file or directory
    if args.dir:
        directory = Path(args.dir)
        if not directory.exists() or not directory.is_dir():
            raise SystemExit(f"디렉터리를 찾을 수 없습니다: {directory}")
        patterns = [p.strip() for p in (args.pattern or "").split(",") if p.strip()]
        image_paths = collect_image_files(directory, patterns)
        if not image_paths:
            raise SystemExit(f"지정한 패턴으로 이미지가 없습니다: {directory} ({args.pattern})")

        print(f"POST {endpoint}")
        print(f"- field: {field_name}")
        print(f"- images: {len(image_paths)}개 from {directory}")
        if extra_data:
            print(f"- extra form fields: {list(extra_data.keys())}")
        if headers:
            print(f"- headers: {list(headers.keys())}")

        with requests.Session() as session:
            if args.bulk:
                # single request with multiple files under the same field name
                files = [build_file_part(field_name, p, override_mime) for p in image_paths]
                resp = session.post(endpoint, files=files, data=extra_data, headers=headers, timeout=args.timeout)
                resp.raise_for_status()
                print("서버 응답:")
                print(resp.text)
            else:
                # multiple requests (one per image)
                for p in image_paths:
                    files = [build_file_part(field_name, p, override_mime)]
                    print(f"-- sending: {p.name}")
                    resp = session.post(endpoint, files=files, data=extra_data, headers=headers, timeout=args.timeout)
                    resp.raise_for_status()
                    print(resp.text)
        return

    # Fallback: single file mode
    if not args.image:
        raise SystemExit("--image 또는 --dir 중 하나는 반드시 지정해야 합니다.")
    image_path = Path(args.image)
    if not image_path.exists() or not image_path.is_file():
        raise SystemExit(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    files = [build_file_part(field_name, image_path, override_mime)]

    print(f"POST {endpoint}")
    print(f"- field: {field_name}")
    print(f"- file: {image_path.name}")
    if extra_data:
        print(f"- extra form fields: {list(extra_data.keys())}")
    if headers:
        print(f"- headers: {list(headers.keys())}")

    response = requests.post(endpoint, files=files, data=extra_data, headers=headers, timeout=args.timeout)
    response.raise_for_status()
    print("서버 응답:")
    print(response.text)


if __name__ == "__main__":
    main()


