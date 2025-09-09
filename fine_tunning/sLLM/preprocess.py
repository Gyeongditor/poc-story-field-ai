import os, json, random, zipfile
from glob import glob
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# NER 모델 불러오기
model_name = "kykim/bert-kor-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=tokenizer, grouped_entities=True)

def extract_character(text):
    results = ner_pipeline(text[:300])  # 긴 본문은 앞부분만 사용
    candidates = [ent['word'] for ent in results if ent['entity_group'] in ["PER", "ANIMAL"]]
    return candidates[0] if candidates else "주인공 없음"

def split_story(text):
    sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    n_pages = min(10, max(5, len(sentences)//2))
    chunk_size = max(2, len(sentences)//n_pages)
    pages = []
    for i in range(n_pages):
        start, end = i * chunk_size, (i + 1) * chunk_size
        page_sentences = sentences[start:end]
        if not page_sentences:
            break
        pages.append(f"{i+1}쪽. " + " ".join(page_sentences))
    return "\n\n".join(pages)

def main():
    # 압축 해제
    zip_path = "1.데이터.zip"
    extract_dir = "dataset"
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

    files = glob(os.path.join(extract_dir, "**/*.json"), recursive=True)
    processed = []

    for f in files:
        with open(f, "r", encoding="utf-8-sig") as infile:
            try:
                data = json.load(infile)
            except:
                continue

        text = data.get("text", "").strip()
        if not text:
            continue

        character = extract_character(text)
        age = random.randint(4, 8)
        sex = random.choice(["남", "여"])
        atmosphere = random.choice(["따뜻한", "신비로운", "모험적인", "즐거운"])
        drawingStyle = random.choice(["수채화", "파스텔", "연필화", "디지털아트"])

        input_text = (
            f"캐릭터: {character}\n"
            f"나이: {age}살\n"
            f"성별: {sex}\n"
            f"분위기: {atmosphere}\n"
            f"그림체: {drawingStyle}\n"
            f"내용: {text[:120]}...\n\n"
            "다음 조건을 반영하여 5~10페이지, 페이지당 2~3문장의 동화를 작성해줘."
        )

        output_text = split_story(text)
        if output_text:
            processed.append({"input": input_text, "output": output_text})

    # 학습/검증 분리
    train, valid = train_test_split(processed, test_size=0.1, random_state=42)

    os.makedirs("processed", exist_ok=True)
    with open("processed/train.jsonl", "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open("processed/valid.jsonl", "w", encoding="utf-8") as f:
        for ex in valid:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f" 전처리 완료: {len(train)} train / {len(valid)} valid")

if __name__ == "__main__":
    main()
