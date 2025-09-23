import argparse
import re
import kss
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

def regex_clean(text: str) -> str:
    text = re.sub(r"(음+|네+|아+|그냥|뭐랄까)", "", text)
    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text)  # 중복 단어 제거
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

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

def normalize_sentences(text: str) -> str:
    sentences = kss.split_sentences(text)
    normalized = []
    for s in sentences:
        s = s.replace("했는데요", "했습니다")
        s = s.replace("갔는데요", "갔습니다")
        s = s.replace("있었는데요", "있었습니다")
        normalized.append(s)
    return " ".join(normalized)

def summarize_text(text: str) -> str:
    if not text.strip():
        return "요약할 내용이 없습니다."

    tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-summarization")
    model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-summarization")

    # 프롬프트 보강
    prompt = (
        "다음은 한 사람이 여러 여행 경험을 이야기한 기록입니다.\n"
        "핵심 장소, 활동, 여행 방식(단체/개인)을 중심으로 짧게 요약해 주세요.\n"
        f"{text}\n\n요약:"
    )

    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
    if inputs.size()[-1] == 0:
        return "요약할 내용이 없습니다."

    summary_ids = model.generate(
        inputs,
        max_length=250,
        min_length=60,
        num_beams=4,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def preprocess_pipeline(raw_text: str) -> str:
    print("[진행] Step1: 규칙 기반 클린업")
    step1 = regex_clean(raw_text)
    if not step1.strip():
        return "클린업 결과가 없습니다."

    print("[진행] Step2: 맞춤법 교정")
    step2 = grammar_correct(step1)
    if not step2.strip():
        step2 = step1

    print("[진행] Step3: 문장 정규화")
    step3 = normalize_sentences(step2)

    print("[진행] Step4: 요약 생성")
    step4 = summarize_text(step3)

    return step4

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
