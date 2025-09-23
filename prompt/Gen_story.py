import argparse
import json
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

def generate_outline(clean_text: str):
    # 간단히 summarization 모델을 이용해 아웃라인 키워드 생성
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-summarization")
    model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-summarization")

    inputs = tokenizer.encode(clean_text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(inputs, max_length=200, min_length=50, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # 간단히 16p 나누기
    pages = []
    sentences = summary.split("다.")  # 문장 분리
    for i in range(16):
        if i < len(sentences):
            text = sentences[i].strip() + "다."
        else:
            text = "여행은 즐거운 모험이었어요."
        pages.append({
            "page": i+1,
            "text": text,
            "scene": f"페이지 {i+1} 장면 설명 (자동 생성 필요)"
        })
    pages[-1]["moral"] = "여행은 우리 마음을 넓혀 준다."
    return pages

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        clean_text = f.read()

    outline = generate_outline(clean_text)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(outline, f, ensure_ascii=False, indent=2)

    print(f"[완료] 아웃라인 저장: {args.output}")
