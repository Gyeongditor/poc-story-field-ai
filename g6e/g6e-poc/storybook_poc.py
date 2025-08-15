import os
import re
import json
from datetime import datetime
from typing import Dict, Any

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from diffusers import AutoPipelineForText2Image
import torch


def read_transcript(transcript_path: str = "transcript.txt") -> str:
    """transcript.txt 읽기"""
    if not os.path.exists(transcript_path):
        return ""
    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content


def build_text_generator():
    """LLaMA 3 8B Instruct 기반 텍스트 생성기 빌드"""
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )


def parse_json_safely(text: str) -> Dict[str, Any]:
    """모델 출력에서 JSON 부분만 안전하게 추출"""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            pass
    return {}


def generate_book_from_transcript(transcript: str) -> Dict[str, Any]:
    """transcript 기반으로 동화 생성"""
    generator = build_text_generator()

    system_instruction = (
        "당신은 아동용 동화 작가이자 이미지 프롬프트 엔지니어입니다. "
        "다음 녹취/키워드를 바탕으로 5페이지 분량의 한국어 동화를 만드세요. "
        "각 페이지는 2~4문장으로 간결하게 쓰고, 각 페이지마다 상세한 삽화 프롬프트를 함께 작성하세요. "
        "또한 동화의 제목과 표지에 어울리는 커버 이미지를 위한 프롬프트도 작성하세요. "
        "반드시 JSON으로만 응답하세요. 다른 텍스트는 금지합니다."
    )

    schema_hint = {
        "title": "string",
        "cover_prompt": "string",
        "pages": [
            {"page": 1, "text": "string", "illustration_prompt": "string"},
            {"page": 2, "text": "string", "illustration_prompt": "string"},
            {"page": 3, "text": "string", "illustration_prompt": "string"},
            {"page": 4, "text": "string", "illustration_prompt": "string"},
            {"page": 5, "text": "string", "illustration_prompt": "string"},
        ],
    }

    prompt = (
        f"{system_instruction}\n\n"
        f"[입력]\n{transcript if transcript else '자유 주제의 따뜻한 모험 동화'}\n\n"
        f"[응답 형식(JSON)]\n{json.dumps(schema_hint, ensure_ascii=False)}\n"
        f"JSON만 출력하세요."
    )

    output = generator(
        prompt,
        max_new_tokens=2048,  # 8K context 지원, 충분히 여유
        do_sample=True,
        temperature=0.8,
        top_p=0.95
    )[0]["generated_text"]

    book = parse_json_safely(output)

    # 리소스 정리
    del generator
    torch.cuda.empty_cache()

    # 최소한의 폴백
    if not book or "pages" not in book:
        book = {
            "title": "마법의 숲에서",
            "cover_prompt": "A whimsical, colorful storybook cover of a magical forest, soft lighting, painterly style",
            "pages": [
                {
                    "page": i + 1,
                    "text": "작은 토끼가 숲에서 모험을 시작해요.",
                    "illustration_prompt": "Cute rabbit in a magical forest, soft light, Studio Ghibli style"
                }
                for i in range(5)
            ],
        }
    return book


def build_image_pipeline():
    """Stable Diffusion 이미지 생성 파이프라인"""
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda")
    try:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
    except Exception:
        pass
    return pipe


def safe_filename(name: str) -> str:
    """파일명 안전하게 변환"""
    name = re.sub(r"[\\/:*?\"<>|]", " ", name).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:60] if len(name) > 60 else name


def save_book_outputs(book: Dict[str, Any], output_root: str = "outputs/storybook") -> str:
    """동화 텍스트와 이미지 저장"""
    os.makedirs(output_root, exist_ok=True)
    title = book.get("title", "storybook")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(output_root, f"{stamp}_{safe_filename(title)}")
    os.makedirs(folder, exist_ok=True)

    # 텍스트 저장
    text_path = os.path.join(folder, "story.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"제목: {title}\n\n")
        for page in book.get("pages", []):
            f.write(f"[Page {page.get('page')}]\n{page.get('text', '')}\n\n")

    # 이미지 생성 및 저장
    pipe = build_image_pipeline()

    # 표지
    cover_prompt = book.get("cover_prompt", "")
    cover_image = pipe(cover_prompt).images[0]
    cover_path = os.path.join(folder, "cover.png")
    cover_image.save(cover_path)

    # 페이지 이미지
    for page in book.get("pages", []):
        page_num = int(page.get("page", 0))
        illu_prompt = page.get("illustration_prompt", "")
        img = pipe(illu_prompt).images[0]
        img_path = os.path.join(folder, f"page_{page_num}.png")
        img.save(img_path)

    del pipe
    torch.cuda.empty_cache()

    return folder


if __name__ == "__main__":
    print("📥 transcript.txt 불러오는 중...")
    transcript = read_transcript("transcript.txt")
    print(transcript[:500] + ("..." if len(transcript) > 500 else ""))  # 미리보기

    print("📖 동화 생성 중...")
    book = generate_book_from_transcript(transcript)
    print(book)  # 디버깅용

    print("🖼️ 표지 및 삽화 생성/저장 중...")
    out_dir = save_book_outputs(book)

    print(f"✅ 완료! 출력 폴더: {out_dir}")
