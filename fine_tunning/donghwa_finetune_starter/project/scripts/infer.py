# scripts/infer.py
import re, json, argparse
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

SYSTEM_PROMPT = (
    "너는 유아동 동화 작가야. 사용자의 JSON 입력(character, age, sex, storyContent, keyword)을 읽고, "
    "한국어로 5쪽 동화를 작성해. 각 쪽은 3~4문장, 온화하고 따뜻한 분위기. "
    "제목은 쓰지 말고 본문만. 꼭 아래 형식으로:\n"
    "### 1쪽\n...문장...\n\n### 2쪽\n...문장...\n\n### 3쪽\n...문장...\n\n### 4쪽\n...문장...\n\n### 5쪽\n...문장...\n"
    "반드시 새로운 내용을 전개하고, 같은 문장을 반복하지 마."
)

# ----- (선택) one-shot 예시로 형식/톤 유도 -----
FEW_SHOT_USER = json.dumps({
    "character": "강아지", "age": 4, "sex": "남",
    "storyContent": "강아지가 공원을 산책하다가 길 잃은 새끼 고양이를 보살펴 줍니다.",
    "keyword": {"atmosphere": "따뜻한", "drawingStyle": "수채화"}
}, ensure_ascii=False)

FEW_SHOT_ASSIST = (
    "### 1쪽\n"
    "강아지는 아침 햇살이 부드럽게 스며드는 공원을 걸었어요. 풀잎에는 맺힌 이슬이 반짝였지요. "
    "그때 작은 울음소리가 들렸어요.\n\n"
    "### 2쪽\n"
    "나뭇가지 아래에서 새끼 고양이가 덜덜 떨고 있었어요. 강아지는 살금살금 다가가 조심스럽게 인사했지요. "
    "고양이는 조그맣게 야옹 하고 대답했어요.\n\n"
    "### 3쪽\n"
    "강아지는 자신의 스카프를 풀어 고양이를 포근하게 감싸 주었어요. 따뜻함이 전해지자 고양이의 눈빛이 조금 밝아졌지요. "
    "둘은 함께 햇살이 드는 벤치로 갔어요.\n\n"
    "### 4쪽\n"
    "벤치에서 강아지는 고양이에게 물을 나눠 주고, 지나가는 새에게 도움을 청했어요. "
    "공원지기가 나타나 잃어버린 고양이를 찾고 있다는 소식을 전했지요.\n\n"
    "### 5쪽\n"
    "곧 주인이 뛰어와 고양이를 안아 올렸어요. 강아지는 안도하며 꼬리를 흔들었지요. "
    "따뜻한 인사가 오가고, 공원에는 다시 평온한 햇살이 번졌어요."
)

def build_messages(user_json_str: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        # one-shot
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSIST},
        # actual input
        {"role": "user", "content": user_json_str},
    ]

PAGE_HEADER_RE = re.compile(r"(?:^|\n)### 1쪽\s*\n", re.MULTILINE)

def extract_story(generated_text: str) -> str:
    """
    프롬프트/메타 텍스트를 날리고, '줄 시작의 ### 1쪽' 이후만 반환.
    """
    m = PAGE_HEADER_RE.search(generated_text)
    if not m:
        return generated_text.strip()
    start = m.start()
    story = generated_text[start:].lstrip()
    return story

# (선택) 후처리: 각 페이지 3~4문장으로 보정
_SENT_SPLIT = re.compile(r"(?<=[.!?。？！]|다\.|요\.|요\?|다\?)\s+")
def _split_sentences(text: str) -> List[str]:
    s = [t.strip() for t in _SENT_SPLIT.split(text.strip()) if t.strip()]
    return s

def rebalance_pages(story: str) -> str:
    pages = []
    chunks = re.split(r"\n\s*###\s*\d쪽\s*\n", story)
    # 첫 split은 빈 조각일 수 있으므로 제거
    chunks = [c for c in chunks if c.strip()]
    for c in chunks[:5]:
        sents = _split_sentences(c)
        # 3~4문장 범위로 자르기/채우기
        if len(sents) < 3:
            # 간단 보정: 마지막 문장에 부가 정보 살짝 덧붙이기
            if sents:
                sents[-1] = sents[-1] + " 작은 마음이 따뜻해졌어요."
            while len(sents) < 3:
                sents.append("이 순간 주인공은 한 번 더 주변을 살폈어요.")
        elif len(sents) > 4:
            sents = sents[:4]
        pages.append(" ".join(sents))

    # 5쪽 맞추기
    while len(pages) < 5:
        pages.append("조용한 숲속에 부드러운 바람이 스며들었어요. 모두의 얼굴에 미소가 번졌지요. 오늘도 따뜻한 하루였어요.")

    out = []
    for i, p in enumerate(pages, 1):
        out.append(f"### {i}쪽\n{p}")
    return "\n\n".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--adapter', type=str, default='outputs/adapter')
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top_p', type=float, default=0.95)
    ap.add_argument('--max_new_tokens', type=int, default=600)
    ap.add_argument('--repetition_penalty', type=float, default=1.15)
    ap.add_argument('--no_repeat_ngram_size', type=int, default=3)
    ap.add_argument('--postprocess', action='store_true')
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter, device_map="auto")
    model.eval()

    # 실제 입력 예시(원하시면 CLI에서 받아도 됩니다)
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

    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tok.eos_token_id,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    story = extract_story(text)
    if args.postprocess:
        story = rebalance_pages(story)
    print(story)

if __name__ == '__main__':
    main()
