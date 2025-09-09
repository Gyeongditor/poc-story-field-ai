from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# 데이터셋 로드
dataset = load_dataset("json", data_files={
    "train": "processed/train.jsonl",
    "validation": "processed/valid.jsonl"
})

# 모델 불러오기
base_model = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

# LoRA 설정
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 토크나이즈
def preprocess(examples):
    return tokenizer(
        examples["input"],
        text_target=examples["output"],
        max_length=1024,
        truncation=True
    )

tokenized = dataset.map(preprocess, batched=True)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=50,
    save_strategy="epoch",
    #evaluation_strategy="epoch",
    eval_steps=500,
    warmup_ratio=0.05,
    bf16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"]
)

if __name__ == "__main__":
    trainer.train()
