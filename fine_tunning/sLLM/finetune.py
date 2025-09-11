from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from peft import LoraConfig, get_peft_model

IGNORE_INDEX = -100

# 1. 데이터셋 로드
dataset = load_dataset("json", data_files={
    "train": "processed/train.jsonl",
    "validation": "processed/valid.jsonl"
})

# 2. 모델 / 토크나이저
base_model = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

# 3. LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Gradient checkpointing & use_cache=False
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

# 4. 전처리 함수 (input+output concat + input 부분 마스킹)
def preprocess(examples):
    sources = [f"<|user|>\n{inp}\n<|assistant|>\n" for inp in examples["input"]]
    targets = [t for t in examples["output"]]

    model_inputs = tokenizer(
        [s + t for s, t in zip(sources, targets)],
        max_length=1024,
        truncation=True
    )

    labels = []
