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
        torch_dtype=preferred_dtype,
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    print("\n=== Generated Story ===\n")
    out = generator(prompt, max_new_tokens=args.max_tokens, temperature=0.8, top_p=0.9, do_sample=True)
    story_text = out[0]["generated_text"][len(prompt):].strip()
    print(story_text)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(story_text)
    print(f"\n동화가 {args.output_file} 에 저장되었습니다.\n")

if __name__ == "__main__":
    main()
