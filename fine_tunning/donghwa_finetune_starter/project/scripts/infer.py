# scripts/infer.py
import re
import json
import argparse
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

SYSTEM_PROMPT = (
    "너는 유아동 동화 작가야. 사용자의 JSON(character, age, sex, storyContent, keyword)을 바탕으로 "
    "완전히 새로운 5쪽 동화를 한국어로 창작해. 각 쪽은 3~4문장. "
    "따뜻하고 포근한 분위기(수채화처럼 부드러운 이미지를 떠올리게), "
    "입력 문장을 그대로 복사하지 말고 새 디테일을 만들어. "
    "출력은 반드시 아래 형식으로만:\n"
    "### 1쪽\n(본문)\n\n### 2쪽\n(본문)\n\n### 3쪽\n(본문)\n\n### 4쪽\n(본문)\n\n### 5쪽\n(본문)\n"
)

# --- one-shot 예시(형식/톤만 유도; 플레이스홀더/메타텍스트 없음) ---
FEW_SHOT_USER = json.dumps({
    "character": "다람쥐", "age": 5, "sex": "여",
    "storyContent": "다람쥐가 가을 숲에서 도토리를 모으다가 길 잃은 무당벌레를 만납니다.",
    "keyword": {"atmosphere": "따뜻한", "drawingStyle": "수채화"}
}, ensure_ascii=False)

FEW_SHOT_ASSIST = (
    "### 1쪽\n"
    "가을 햇살이 숲에 비치자 다람쥐의 꼬리도 금빛으로 반짝였어요. 바삭바삭 낙엽 소리를 들으며 도토리를 주웠지요. "
    "그때 아주 작은 목소리가 귓가를 간질였어요.\n\n"
    "### 2쪽\n"
    "낙엽 사이에서 무당벌레 한 마리가 떨고 있었어요. 다람쥐는 조심스레 손바닥을 내밀어, 햇살이 드는 자리로 옮겨 주었지요. "
    "무당벌레는 점박이 날개를 살짝 펴 보이며 고개를 끄덕였어요.\n\n"
    "### 3쪽\n"
    "둘은 함께 숲길을 걸으며 길 표식을 찾아보았어요. 도토리 향기와 나무의 숨결이 포근하게 감싸 안았지요. "
    "다람쥐는 작은 잎배를 만들어 개울을 건너도록 도와주었어요.\n\n"
    "### 4쪽\n"
    "노란 이파리 아래, 무당벌레의 집이 가까워졌어요. 길가의 버섯 우산들이 다리를 만들어 주는 듯 보였지요. "
    "마침내 고목의 껍질 틈에서 반짝이는 무당벌레 가족의 집을 찾았어요.\n\n"
    "### 5쪽\n"
    "무당벌레 가족은 기쁨의 점춤을 추며 인사했어요. 다람쥐의 마음도 따뜻하게 데워졌지요. "
    "저녁 바람이 살짝 불어오자, 숲은 부드러운 수채화처럼 고요히 물들었어요."
)

def build_messages(user_json_str: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSIST},
        {"role": "user", "content": user_json_str},
    ]

# "가장 마지막" ### 1쪽 이후만 추출 (템플릿/예시 오염 방지)
PAGE1_RE = re.compile(r"(?:^|\n)###\s*1쪽\s*\n", re.MULTILINE)

def extract_story(text: str) -> str:
    matches = list(PAGE1_RE.finditer(text))
    if not matches:
        return text.strip()
    start = matches[-1].start()
    story = text[start:].lstrip()
    # user/assistant 같은 메타라인 제거
    story = "\n".join([ln for ln in story.splitlines() if not re.match(r"^\s*(user|assistant)\s*$", ln, re.I)])
    return story

# --------- 문장 분리(lookbehind 없음, 캡처/재조립) ----------
_END_TOKENS = r"[.!?。？！]|다\.|요\.|요\?|다\?"
_SENT_SPLIT = re.compile(rf"({_END_TOKENS})\s+")

def count_sents(par: str) -> int:
    parts = _SENT_SPLIT.split(par.strip())
    sents, i = [], 0
    while i < len(parts):
        chunk = (parts[i] or "").strip()
        sep = parts[i+1] if i+1 < len(parts) else ""
        if chunk:
            sents.append((chunk + (sep or "")).strip())
        i += 2
    return len([s for s in sents if s])

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

    # 토크나이저/모델 로드
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # pad 없으면 eos로 통일

    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter, device_map="auto")
    model.eval()

    # eos를 '정수 1개'로 강제
    eos_id = tok.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0]
    if eos_id is None:
        eos_id = tok.convert_tokens_to_ids(tok.eos_token) if tok.eos_token else tok.pad_token_id
    assert isinstance(eos_id, int), f"eos_token_id must be int, got: {type(eos_id)}"

    # 실제 입력 (필요 시 argparse로 교체 가능)
    sample = {
      "character": "토끼",
      "age": 5,
      "sex": "여",
      "storyContent": "토끼가 숲에서 친구를 만났습니다.",
      "keyword": {"atmosphere": "따뜻한", "drawingStyle": "수채화"}
    }
    user_json = json.dumps(sample, ensure_ascii=False)

    messages = build_messages(user_json)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 금지어: 플레이스홀더/메타 문자열 차단
    bad_words = ["...문장...", "user", "assistant"]
    bad_words_ids = [tok(bw, add_special_tokens=False).input_ids for bw in bad_words]

    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,   # ✅ 공식 인자로 최소 길이 보장
            eos_token_id=eos_id,
            pad_token_id=eos_id,                  # pad/eos 통일(마스킹/디바이스 이슈 예방)
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    story = extract_story(text)

    # 간단 검증: 각 페이지 3+문장 권장 (경고만)
    pages = re.split(r"\n\s*###\s*\d쪽\s*\n", story)
    if len([p for p in pages if p.strip()]) < 5:
        print("[WARN] 5쪽 형식이 완전하지 않습니다. 프롬프트/하이퍼파라미터를 조정하세요.")
    else:
        for i, p in enumerate([x for x in pages if x.strip()][:5], 1):
            if count_sents(p) < 3:
                print(f"[WARN] {i}쪽 문장 수가 적습니다. temperature/top_p를 높이거나 min_new_tokens를 늘려보세요.")

    print(story)

if __name__ == '__main__':
    main()
