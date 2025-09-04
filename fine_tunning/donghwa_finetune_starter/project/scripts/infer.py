
import json, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

SYSTEM_PROMPT = (
    "너는 유아동 동화 작가야. 사용자의 JSON 입력을 읽고, 한국어로 5쪽 동화를 작성해. "
    "각 쪽은 3~4문장, 온화한 어조(요/다체 혼용 가능). 제목은 생략하고 본문만 출력. "
    "페이지 표시는 '### 1쪽'처럼 달고, 내용 외 불필요한 설명은 쓰지 마."
)

def build_messages(user_json_str: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_json_str},
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--adapter', type=str, default='outputs/adapter')
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--max_new_tokens', type=int, default=600)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter, device_map="auto")
    model.eval()

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
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    gen = text.split("### 1쪽", 1)
    if len(gen) > 1:
        print("### 1쪽" + gen[1])
    else:
        print(text)

if __name__ == '__main__':
    main()
