import argparse
import re
import kss
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# ------------------------
# 1) 규칙 기반 클린업
# ------------------------
def regex_clean(text: str) -> str:
    # 추임새/불필요한 구어체 제거
    text = re.sub(r"(음+|네+|아+|그냥|뭐랄까)", "", text)
    # 중복 단어 제거 (예: "여행 여행")
    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text)
    # 불필요 공백 제거
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# ------------------------
# 2) 맞춤법 교정 (py-hanspell 기반)
# ------------------------
def grammar_correct(text: str) -> str:
    try:
        from hanspell import spell_checker
        result = spell_checker.check(text)
        checked = result.checked.strip()
        if not checked:
            return text
        return checked
    except Exception as e:
        print(f"[경고] 맞춤법 교정 실패 → 원문 사용: {e}")
        return text


# ------------------------
# 3) 요약 (KoBART Summarization)
# ------------------------
def summarize_text(text: str) -> str:
    if not text.strip():
        return "요약할 내용이 없습니다."

    tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-summarization")
    model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-summarization")

    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    if inputs.size()[-1] == 0:
        return "요약할 내용이 없습니다."

    summary_ids = model.generate(
        inputs,
        max_length=200,
        min_length=50,
        num_beams=4,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# ------------------------
# 4) 전체 파이프라인
# ------------------------
def preprocess_pipeline(raw_text: str) -> str:
    print("[진행] Step1: 규칙 기반 클린업")
    step1 = regex_clean(raw_text)
    if not step1.strip():
        return "클린업 결과가 없습니다."

    print("[진행] Step2: 맞춤법 교정")
    step2 = grammar_correct(step1)
    if not step2.strip():
        step2 = step1

    print("[진행] Step3: 요약 생성")
    step3 = summarize_text(step2)

    return step3


# ------------------------
# 실행 진입점
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="STT 원본 텍스트 파일")
    parser.add_argument("--output", required=True, help="클린/요약 텍스트 출력 파일")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        raw_text = f.read()

    clean_text = preprocess_pipeline(raw_text)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(clean_text)

    print(f"[완료] clean text 저장 → {args.output}")
