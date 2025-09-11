import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="요약할 텍스트 파일")
    parser.add_argument("--prompt_file", default="summarize_prompt.txt", help="프롬프트 템플릿")
    parser.add_argument("--output_file", default="summary_output.txt", help="요약 결과 저장 파일")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct", help="허깅페이스 모델 ID")
    parser.add_argument("--max_tokens", type=int, default=500)
    args = parser.parse_args()

    raw = open(args.input_file, encoding="utf-8").read()
    template = load_template(args.prompt_file)
    prompt = template.format(content=raw)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype="bfloat16"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

    print("\n=== Generating Summary ===\n")
    out = pipe(prompt, max_new_tokens=args.max_tokens, temperature=0.3, top_p=0.9)
    summary_text = out[0]["generated_text"][len(prompt):].strip()

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"요약 결과가 {args.output_file} 에 저장되었습니다.")

if __name__ == "__main__":
    main()
