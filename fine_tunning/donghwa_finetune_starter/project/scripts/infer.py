# scripts/infer.py
import re
import json
import argparse
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

SYSTEM_PROMPT = (
    "너는 유아동 동화 작가야. 사용자의 JSON(character, age, sex, storyContent, keyword)을 바탕으로 "
    "완전히 새로운 5쪽 동화를 한국어로 창작해. 각 쪽은 3~4문장. "
    "따뜻하고 포근한 분위기(수채화처럼 부드러운 이미지를 떠올리게). "
    "입력 문장을 그대로 복사하지 말고, 사건과 묘사를 확장해 새 디테일을 만들어. "
    "출력은 정확히 아래 형식으로만:\n"
    "### 1쪽\n(본문)\n\n### 2쪽\n(본문)\n\n### 3쪽\n(본문)\n\n### 4쪽\n(본문)\n\n### 5쪽\n(본문)\n"
    "주의: 코드블록, 영어 메타단어(user/assistant/system/tool_call), 따옴표/중괄호가 섞인 JSON 에코, "
    "‘...문장...’, ‘생략’, ‘略’, ‘keeppage’ 같은 단어는 절대 출력하지 마."
)

# ---------- 포맷 유틸 ----------
PAGE1_RE = re.compile(r"(?:^|\n)###\s*1쪽\s*\n", re.MULTILINE)
HEADER_RE = re.compile(r"^###\s*(\d)쪽\s*$")
END_TOKENS = r"[.!?。？！]|다\.|요\.|요\?|다\?"
SENT_SPLIT = re.compile(rf"({END_TOKENS})\s+")

FORBIDDEN_LINE_RE = re.compile(
    r"^(?:\s*```|.*\b(user|assistant|system|tool_call)\b.*|.*\.\.\.문장\.\.\..*|.*생략.*|.*略.*|.*keeppage.*|.*<.*>.*)$",
    re.IGNORECASE,
)

def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = SENT_SPLIT.split(text.strip())
    sents = []
    i = 0
    while i < len(parts):
        chunk = (parts[i] or "").strip()
        sep = parts[i+1] if i+1 < len(parts) else ""
        if chunk:
            sents.append((chunk + (sep or "")).strip())
        i += 2
    return [s for s in sents if s]

def extract_after_last_page1(txt: str) -> str:
    """가장 마지막 '### 1쪽'부터 끝까지. 앞쪽 템플릿/에코 제거."""
    matches = list(PAGE1_RE.finditer(txt))
    if not matches:
        return txt.strip()
    start = matches[-1].start()
    out = txt[start:].lstrip()
    # 명백한 메타/쓰레기 라인 제거
    lines = []
    for ln in out.splitlines():
        if FORBIDDEN_LINE_RE.match(ln):
            continue
        # 괄호투성이 JSON 에코/깨진 따옴표 라인 제거
        if ln.count("{") + ln.count("}") + ln.count("\"") + ln.count("'") > 6:
            continue
        lines.append(ln)
    return "\n".join(lines).strip()

def parse_pages(text: str) -> List[Tuple[int, str]]:
    """### n쪽 헤더 기준으로 페이지 추출."""
    pages = []
    current_idx = None
    buf = []
    for ln in text.splitlines():
        m = HEADER_RE.match(ln.strip())
        if m:
            # flush
            if current_idx is not None:
                pages.append((current_idx, "\n".join(buf).strip()))
                buf = []
            current_idx = int(m.group(1))
        else:
            if current_idx is not None:
                buf.append(ln)
    if current_idx is not None:
        pages.append((current_idx, "\n".join(buf).strip()))
    return pages

def clean_paragraph(par: str) -> str:
    # 과도한 공백/잡기호 정리
    par = re.sub(r"\s+", " ", par).strip()
    # 남은 메타/금지어 제거(소프트)
    par = re.sub(r"(user|assistant|system|tool_call)", "", par, flags=re.I)
    par = par.replace("...문장...", "").replace("생략", "").replace("略", "").replace("keeppage", "")
    return par.strip()

def enforce_page_rules(pages: List[Tuple[int, str]]) -> List[str]:
    """5쪽 보장 + 각 3~4문장 범위로 유도(하드 컷/보정 없음; 지나치게 짧으면 경고만)."""
    out = [""] * 5
    for idx, par in pages:
        if 1 <= idx <= 5:
            out[idx-1] = clean_paragraph(par)
    # 부족한 페이지는 간단한 문장으로 채워 형식 보장(의미 변형 최소화)
    filler = "부드러운 바람이 숲을 스치자 마음까지 따뜻해졌어요."
    for i in range(5):
        if not out[i]:
            out[i] = filler
    return out

def format_story(pages: List[str]) -> str:
    blocks = []
    for i, p in enumerate(pages, 1):
        blocks.append(f"### {i}쪽\n{p}")
    return "\n\n".join(blocks)

# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--adapter', type=str, default='outputs/adapter')
    ap.add_argument('--temperature', type=float, default=0.85)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--top_k', type=int, default=50)
    ap.add_argument('--max_new_tokens', type=int, default=700)
    ap.add_argument('--min_new_tokens', type=int, default=300)
    ap.add_argument('--repetition_penalty', type=float, default=1.2)
    ap.add_argument('--no_repeat_ngram_size', type=int, default=4)
    args = ap.parse_args()

    # 토크나이저/모델
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter, device_map="auto")
    model.eval()

    # eos를 정수 1개로 통일
    eos_id = tok.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0]
    if eos_id is None:
        eos_id = tok.convert_tokens_to_ids(tok.eos_token) if tok.eos_token else tok.pad_token_id

    # 실제 입력 (필요시 argparse로 받아도 됨)
    sample = {
        "character": "토끼",
        "age": 5,
        "sex": "여",
        "storyContent": "토끼가 숲에서 친구를 만났습니다.",
        "keyword": {"atmosphere": "따뜻한", "drawingStyle": "수채화"}
    }
    user_json = json.dumps(sample, ensure_ascii=False)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_json},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 금지어: 플레이스홀더/메타/잡문자 후보
    bad_words = [
        "...문장...", "user", "assistant", "system", "tool_call",
        "생략", "略", "keeppage",
        "```", "<tool_call>", "<", ">", "页面", "acistsystem"
    ]
    bad_words_ids = [tok(bw, add_special_tokens=False).input_ids for bw in bad_words if bw]

    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,   # 커스텀 processor 없이 최소 길이 보장
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
        )

    text = tok.decode(out[0], skip_special_tokens=True)

    # 1) 프롬프트/예시 오염 제거
    story_raw = extract_after_last_page1(text)

    # 2) 페이지 파싱 → 형식 보장
    parsed = parse_pages(story_raw)
    pages = enforce_page_rules(parsed)

    # 3) (소프트) 페이지별 문장 수 점검
    for i, p in enumerate(pages, 1):
        n = len(split_sentences(p))
        if n < 3:
            # 짧으면 힌트 메시지만 찍고, 내용은 변경하지 않음(하드 보정은 원치 않음)
            print(f"[WARN] {i}쪽이 {n}문장입니다. temperature/top_p/길이를 조정해보세요.")

    print(format_story(pages))

if __name__ == '__main__':
    main()
