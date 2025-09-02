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

---

## LoRA 파인튜닝 (AWS g6e.xlarge)

아래 스크립트는 `hf_instruction_*.jsonl` 데이터셋으로 QLoRA 기반 SFT를 수행합니다.

### 1) 환경 준비 (Ubuntu 22.04 on g6e.xlarge, A10G 24GB)

- CUDA 드라이버와 nvidia-container-toolkit이 설치된 딥러닝 AMI 권장
- Python 3.10+ 권장

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-train.txt
```

주의: Windows 로컬에서는 `bitsandbytes`가 제한됨. g6e 인스턴스 내에서 설치/실행 권장.

### 2) 데이터 준비

`convert_dataset.py`로 이미 `story/hf_instruction_*.jsonl`이 생성된 상태라고 가정합니다.
다른 경로에 있다면 `--data_dir`로 지정하세요.

### 3) 학습 실행 예시

```bash
python train_lora.py \
  --model_name_or_path beomi/llama-2-ko-7b \
  --data_dir ./story \
  --train_file hf_instruction_train.jsonl \
  --eval_file hf_instruction_val.jsonl \
  --output_dir ./lora-out \
  --max_seq_len 2048 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --quantize_4bit true --bf16 true --use_gradient_checkpointing true
```

메모리 팁(g6e.xlarge, 24GB GPU):
- `per_device_train_batch_size=1`, `grad_accum=16~32`, `bf16=true`, `4bit=true`
- 컨텍스트 길이가 길면 `max_seq_len`을 1536/1024로 줄이세요.

### 3-1) 포그라운드 실행 (터미널에서 바로 보기)

```bash
python train_lora.py \
  --model_name_or_path beomi/llama-2-ko-7b \
  --data_dir ./story \
  --train_file hf_instruction_train.jsonl \
  --eval_file hf_instruction_val.jsonl \
  --output_dir ./lora-out \
  --max_seq_len 2048 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --quantize_4bit true --bf16 false --fp16 true --use_gradient_checkpointing true
```

- 출력 로그가 터미널에 실시간 표시됩니다. 다른 창에서 GPU 사용률 확인:
```bash
watch -n 1 nvidia-smi
```
- 중단 후 재개: 동일한 `--output_dir`로 다시 실행하면 자동으로 이어서 학습합니다.

### 4) 추론 (LoRA 어댑터 로드)

```bash
python infer_lora.py \
  --base_model beomi/llama-2-ko-7b \
  --adapter_dir ./lora-out \
  --title "해님달님" \
  --character "해님과 달님" \
  --age 7 \
  --sex "-" \
  --atmosphere "따뜻한 분위기" \
  --drawingStyle "수채화풍" \
  --storyContent "빨리 달리는 토끼와 느리지만 꾸준한 거북이가 경주를 벌인다." \
  --page 1 --sentences 3 --words 40
```

자유 입력 문자열을 직접 넘기려면 `--input` 옵션 하나로 전달하면 됩니다.

### 5) 기타 모델 선택 팁
- 한국어 성능: `Qwen2-7B-Instruct`, `Llama-3.1-8B-Instruct`, `Mistral-Nemo-Instruct`
- VRAM 여유가 부족하면 `Qwen2.5-7B-Instruct-GPTQ-Int4` 같은 4bit 가중치를 활용