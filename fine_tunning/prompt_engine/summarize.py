import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_template(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def clean_text(text: str) -> str:
    """
    STT나 대화 텍스트에서 흔히 등장하는 불필요한 추임새, 반복 어구, 
    과도한 공백을 제거해 모델 입력을 간결화합니다.
    """
    # 1) '네', '음', '아', '어' 등 단독 추임새 제거
    text = re.sub(r'\b(네|음|아|어)\b[.,]?\s*', '', text)

    # 2) 2회 이상 반복되는 단어/문장부호 압축 (예: "ㅋㅋㅋㅋ", "……")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # 3) 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="요약할 원문 텍스트 파일")
    parser.add_argument("--prompt_file", default="summarize_prompt.txt", help="프롬프트 템플릿")
    parser.add_argument("--output_file", default="summary_output.txt", help="요약 결과 저장 파일")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct", help="허깅페이스 모델 ID")
    parser.add_argument("--max_tokens", type=int, default=400, help="최대 생성 토큰 수")
    args = parser.parse_args()

    # 입력 텍스트 읽기 & 전처리
    raw = open(args.input_file, encoding="utf-8").read()
    cleaned = clean_text(raw)

    # 프롬프트 로드 및 삽입
    template = load_template(args.prompt_file)
    prompt = template.format(content=cleaned)

    # 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    preferred_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        dtype=preferred_dtype
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    print("\n=== Generating Summary ===\n")
    out = pipe(
        prompt,
        max_new_tokens=args.max_tokens,
        temperature=0.6,       # 낮은 온도: 안정된 결과
        top_p=0.9,
        repetition_penalty=1.3,
        do_sample=False
    )

    # 프롬프트 부분을 제거하고 생성된 결과만 저장
    summary_text = out[0]["generated_text"][len(prompt):].strip()

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n요약 결과가 {args.output_file} 에 저장되었습니다.\n")

if __name__ == "__main__":
    main()
