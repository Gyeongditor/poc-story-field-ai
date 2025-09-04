# scripts/infer.py
import re
import json
import argparse
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# ---------------- 공통 유틸 ----------------
END_TOKENS = r"[.!?。？！]|다\.|요\.|요\?|다\?"
SENT_SPLIT = re.compile(rf"({END_TOKENS})\s+")
HEADER_RE = re.compile(r"^###\s*(\d)쪽\s*$")

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

def clean_line_noise(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    # 메타/플레이스홀더/다국어 잡문자 제거
    forb = ["...문장...", "user", "assistant", "system", "tool_call",
            "생략", "略", "keeppage", "(본문)", "(비워둠)", "페이지", "页面", "página"]
    for w in forb:
        s = s.replace(w, "")
    # 과한 괄호/따옴표만 있는 줄 제거
    if (s.count("{") + s.count("}") + s.count("\"") + s.count("'")) > 6:
        s = ""
    return s.strip()

def ensure_k_sentences(par: str, kmin=3, kmax=4) -> Tuple[str, bool]:
    sents = split_sentences(par)
    if len(sents) < kmin:
        return par, False
    if len(sents) > kmax:
        sents = sents[:kmax]
    return " ".join(sents), True

def format_story(pages: List[str]) -> str:
    out = []
    for i, p in enumerate(pages, 1):
        out.append(f"### {i}쪽\n{p.strip()}")
    return "\n\n".join(out)

# ---------------- 프롬프트 ----------------
SYS_OUTLINE = (
    "너는 유아동 동화 작가야. 사용자의 JSON(character, age, sex, storyContent, keyword)을 읽고, "
    "한국어로 5쪽 동화를 만들기 위한 **아웃라인 5개**를 먼저 작성해.\n"
    "- 각 항목은 한 문장으로, 사건의 핵심만 간단히 적기.\n"
    "- 따뜻하고 포근한 분위기, 수채화 같은 이미지.\n"
    "- 입력 문장을 복사하지 말고 새 디테일을 만들어.\n"
    "- 출력은 번호만: 1) … 2) … 3) … 4) … 5) …\n"
    "- 다른 텍스트(코드블록, 메타 단어, JSON 등) 금지."
)

SYS_WRITE_PAGE = (
    "너는 유아동 동화 작가야. 아래 정보와 주어진 '아웃라인 한 항목'을 바탕으로 "
    "한국어 본문을 3~4문장으로 써라. 따뜻하고 포근한 분위기(수채화 같은 이미지), "
    "입력 문장 복사 금지, 새 디테일을 만들어라. 오직 본문만 출력(제목/번호/코드블록 금지)."
)

# ---------------- 생성 함수 ----------------
def generate_text(model, tok, prompt, bad_words_ids, max_new, min_new, temperature, top_p, top_k, eos_id):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            bad_words_ids=bad_words_ids,
        )
    text = tok.decode(out[0], skip_special_tokens=True).strip()
    return text

def extract_outline_items(text: str) -> List[str]:
    # "1) ... 2) ... 3) ... 4) ... 5) ..." 혹은 줄바꿈 번호 패턴 허용
    items = []
    # 줄바꿈 기준
    for ln in text.splitlines():
        ln = clean_line_noise(ln)
        m = re.match(r"^\s*(?:\d+[\).\-\:]|\-)\s*(.+)$", ln)
        if m:
            items.append(m.group(1).strip())
    # 한 줄에 모두 쓸 수도 있으므로 보조 파싱
    if not items:
        parts = re.split(r"\s*\d+[\).\-\:]\s*", text)
        parts = [clean_line_noise(p) for p in parts if clean_line_noise(p)]
        if parts:
            items = parts[:5]
    return items[:5]

# ---------------- 메인 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--adapter', type=str, default='outputs/adapter')
    ap.add_argument('--temperature', type=float, default=0.85)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--top_k', type=int, default=50)
    ap.add_argument('--max_new_tokens', type=int, default=220)   # 페이지/아웃라인 단위라 짧게
    ap.add_argument('--min_new_tokens', type=int, default=80)
    ap.add_argument('--retries', type=int, default=3)
    args = ap.parse_args()

    # 모델 로드
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter, device_map="auto")
    model.eval()

    # eos 정수 1개 보장
    eos_id = tok.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0]
    if eos_id is None:
        eos_id = tok.convert_tokens_to_ids(tok.eos_token) if tok.eos_token else tok.pad_token_id

    # 금지어
    bad_words = [
        "...문장...", "user", "assistant", "system", "tool_call",
        "생략", "略", "keeppage", "```", "<", ">", "(본문)", "(비워둠)", "페이지", "页面", "página"
    ]
    bad_words_ids = [tok(w, add_special_tokens=False).input_ids for w in bad_words if w]

    # === 입력 예시(원하면 argparse로 받아도 됨) ===
    user = {
        "character": "토끼",
        "age": 5,
        "sex": "여",
        "storyContent": "토끼가 숲에서 친구를 만났습니다.",
        "keyword": {"atmosphere": "따뜻한", "drawingStyle": "수채화"}
    }
    user_json = json.dumps(user, ensure_ascii=False)

    # 1) 아웃라인 생성
    outline_prompt = (
        f"<|system|>\n{SYS_OUTLINE}\n</s>\n"
        f"<|user|>\n{user_json}\n</s>\n"
        f"<|assistant|>\n"
    )
    outline_text = generate_text(
        model, tok, outline_prompt, bad_words_ids,
        max_new=args.max_new_tokens, min_new=50,
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, eos_id=eos_id
    )
    items = extract_outline_items(outline_text)
    if len(items) < 5:
        # 아웃라인이 부족하면 간단히 채우기
        while len(items) < 5:
            items.append("주인공이 새로운 감정을 배우고 친구와 함께 문제를 해결한다.")

    # 2) 페이지별 본문 생성(검증/재시도)
    pages = []
    for i in range(5):
        outline_item = items[i]
        ok = False
        text_ok = ""
        for attempt in range(args.retries):
            write_prompt = (
                f"<|system|>\n{SYS_WRITE_PAGE}\n</s>\n"
                f"<|user|>\n"
                f"[입력 JSON]\n{user_json}\n\n"
                f"[이번 페이지 아웃라인]\n- {outline_item}\n\n"
                f"[요구 형식]\n- 문장만 출력 (3~4문장)\n- 번호/헤더/코드블록 금지\n</s>\n"
                f"<|assistant|>\n"
            )
            body = generate_text(
                model, tok, write_prompt, bad_words_ids,
                max_new=args.max_new_tokens, min_new=args.min_new_tokens,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, eos_id=eos_id
            )
            body = clean_line_noise(body)
            body, passed = ensure_k_sentences(body, 3, 4)
            if passed and body:
                text_ok = body
                ok = True
                break
        if not ok:
            # 최종 실패시 간단한 안전 문단
            text_ok = "부드러운 바람이 숲을 스치자 마음이 포근해졌어요. 작은 친절이 오늘의 기적이 되었지요. 모두의 눈빛이 따뜻하게 빛났어요."
        pages.append(text_ok)

    # 3) 포맷팅 출력
    print(format_story(pages))

if __name__ == '__main__':
    main()
