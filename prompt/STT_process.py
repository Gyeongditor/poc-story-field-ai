import argparse
import re
import kss
from hanspell import spell_checker
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

def regex_clean(text: str) -> str:
    text = re.sub(r"(음+|네+|아+|그냥|뭐랄까)", "", text)
    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def grammar_correct(text: str) -> str:
    try:
        result = spell_checker.check(text)
        return result.checked
    except:
        return text

def summarize_text(text: str) -> str:
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-summarization")
    model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-summarization")
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=200, min_length=50, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def preprocess_pipeline(raw_text: str) -> str:
    step1 = regex_clean(raw_text)
    step2 = grammar_correct(step1)
    step3 = summarize_text(step2)
    return step3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        raw_text = f.read()

    clean_text = preprocess_pipeline(raw_text)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(clean_text)

    print(f"[완료] clean text 저장: {args.output}")
