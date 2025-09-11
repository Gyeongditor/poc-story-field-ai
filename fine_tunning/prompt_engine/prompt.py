import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_prompt_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_prompt(template: str, keywords: str, character: str, summary: str) -> str:
    return template.format(keywords=keywords, character=character, summary=summary)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", required=True, help="동화 분위기 키워드")
    parser.add_argument("--character", required=True, help="주인공 정보")
    parser.add_argument("--summary", help="이야기 줄거리 직접 입력")
    parser.add_argument("--summary_file", help="summarize.py 결과 파일 경로")
    parser.add_argument("--prompt_file", default="story_prompt.txt", help="동화 프롬프트 템플릿")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--output_file", default="story_output.txt", help="생성된 동화 저장 파일 경로")
    args = parser.parse_args()

    # summary 선택: 파일이 있으면 우선
    if args.summary_file:
        with open(args.summary_file, "r", encoding="utf-8") as f:
            summary_text = f.read().strip()
    elif args.summary:
        summary_text = args.summary
    else:
        parser.error("요약 텍스트(--summary)나 요약 파일(--summary_file) 중 하나는 반드시 필요합니다.")

    template = load_prompt_template(args.prompt_file)
    prompt = build_prompt(template, args.keywords, args.character, summary_text)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    preferred_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        dtype=preferred_dtype,
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    print("\n=== Generated Story ===\n")
    # 일부 모델은 generation_config 방식만 허용. 경고 제거를 위해 기본 인자만 사용
    out = generator(
        prompt,
        max_new_tokens=args.max_tokens,
    )
    story_text = out[0]["generated_text"][len(prompt):].strip()
    # [Page] 이전의 프리앰블 제거
    first_page_idx = story_text.find("[Page ")
    if first_page_idx > 0:
        story_text = story_text[first_page_idx:]

    import re as _re
    # 1) [Page N]이 문장과 같은 줄에 붙은 경우 분리
    story_text = _re.sub(r"\s*\[Page\s*(\d+)\]\s*", lambda m: f"\n[Page {m.group(1)}]\n", story_text)
    story_text = story_text.strip()

    # 2) 라인 정리 및 한국어 비율 필터
    lines = [ln.strip() for ln in story_text.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        if ln.startswith('[Page '):
            cleaned.append(ln)
            continue
        num_ko = sum(1 for ch in ln if '가' <= ch <= '힣')
        if num_ko >= max(1, len(ln) // 3):
            cleaned.append(ln)

    # 3) 페이지 구조 강제: [Page 1]~[Page 8], 각 5문장 이하로 수집
    pages_map = {}
    current = None
    for ln in cleaned:
        m = _re.match(r"^\[Page\s*(\d+)\]$", ln)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 8:
                current = n
                pages_map.setdefault(n, [])
            else:
                current = None
            continue
        if current is not None and len(pages_map[current]) < 5:
            pages_map[current].append(ln)

    # 4) [Page]가 전혀 없으면 5문장씩 자동 분할
    if not pages_map:
        sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", story_text) if s.strip()]
        page_num = 1
        idx = 0
        while idx < len(sentences) and page_num <= 8:
            pages_map[page_num] = sentences[idx:idx+5]
            idx += 5
            page_num += 1

    # 5) 페이지 조립 (최대 8페이지, 각 최대 5문장)
    assembled = []
    for n in sorted(pages_map.keys()):
        if n < 1 or n > 8:
            continue
        if not pages_map[n]:
            continue
        content = pages_map[n][:5]
        assembled.append(f"[Page {n}]")
        assembled.extend(content)
    story_text = "\n".join(assembled).strip()
    print(story_text)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(story_text)
    print(f"\n동화가 {args.output_file} 에 저장되었습니다.\n")

if __name__ == "__main__":
    main()
