import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_prompt_template(path: str) -> str:
    """프롬프트 템플릿 파일 읽기"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_prompt(template: str, keywords: str, character: str, summary: str) -> str:
    """프롬프트 템플릿 채우기"""
    return template.format(
        keywords=keywords,
        character=character,
        summary=summary
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", type=str, required=True, help="동화 분위기 키워드")
    parser.add_argument("--character", type=str, required=True, help="주인공 정보 (이름, 나이, 성별 등)")
    parser.add_argument("--summary", type=str, required=True, help="이야기 줄거리")
    parser.add_argument("--prompt_file", type=str, default="story_prompt.txt", help="프롬프트 템플릿 경로")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="허깅페이스 모델 ID")
    parser.add_argument("--max_tokens", type=int, default=2000, help="최대 생성 토큰 수")
    args = parser.parse_args()

    # 프롬프트 불러오기
    template = load_prompt_template(args.prompt_file)
    prompt = build_prompt(template, args.keywords, args.character, args.summary)

    # 모델 로드
    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype="bfloat16"
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    # 생성
    print("\n=== Generated Story ===\n")
    output = generator(
        prompt,
        max_new_tokens=args.max_tokens,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )

    print(output[0]["generated_text"])

if __name__ == "__main__":
    main()
