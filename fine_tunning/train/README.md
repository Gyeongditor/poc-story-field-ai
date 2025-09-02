# 동화 줄거리 생성 파인튜닝 데이터

이 폴더에는 동화 줄거리 생성 데이터만으로 구축된 HuggingFace 및 OpenAI GPT 파인튜닝용 데이터가 포함되어 있습니다.

## 데이터 구조
- story/: 동화 줄거리(텍스트) 기반 데이터
  - hf_instruction_*.jsonl: instruction-tuning (Alpaca 스타일: instruction/input/output)
  - openai_chat_*.jsonl: OpenAI 채팅 파인튜닝 포맷(messages 배열)
  - hf_causal_*.jsonl: causal LM 포맷({"text"})

## 데이터 개수
- Training: 1,603개 동화 파일 → 각 포맷별 샘플 수 생성
- Validation: 216개 동화 파일 → 각 포맷별 샘플 수 생성
- 각 파일의 paragraph 수만큼 instruction 샘플 생성
- 각 파일당 1개의 causal LM 샘플 생성

## HuggingFace 예시
Python에서 datasets 로드 예시:

```python
from datasets import load_dataset
ds = load_dataset("json", data_files={"train": "story/hf_instruction_train.jsonl", "validation": "story/hf_instruction_val.jsonl"})
print(ds)
```

Causal LM 예시(Transformers Trainer):

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

ds = load_dataset("json", data_files={"train": "story/hf_causal_train.jsonl", "validation": "story/hf_causal_val.jsonl"})
tok = AutoTokenizer.from_pretrained("gpt2")
def tok_fn(batch):
    return tok(batch["text"], truncation=True)
ds_tok = ds.map(tok_fn, batched=True, remove_columns=ds["train"].column_names)
model = AutoModelForCausalLM.from_pretrained("gpt2")
args = TrainingArguments(output_dir="./out", per_device_train_batch_size=2, num_train_epochs=1)
trainer = Trainer(model=model, args=args, train_dataset=ds_tok["train"], eval_dataset=ds_tok["validation"])
trainer.train()
```

## OpenAI 파인튜닝 예시
- story/openai_chat_train.jsonl 파일을 업로드하여 사용하세요.
- OpenAI CLI 예시 (모델 이름과 API 사용법은 최신 문서를 참고):

```bash
openai files upload --purpose fine-tune --file story/openai_chat_train.jsonl
openai files upload --purpose fine-tune --file story/openai_chat_val.jsonl
# 이후 fine-tuning job 생성
# openai fine_tuning.jobs.create -t <TRAIN_FILE_ID> -v <VAL_FILE_ID> -m gpt-4o-mini
```

## 특징
- 각 동화의 메타데이터(제목, 분류, 연령)를 포함한 풍부한 컨텍스트
- 페이지별 세부 정보와 함께한 구조화된 학습 데이터
- 한국어 동화 생성에 특화된 고품질 데이터셋
