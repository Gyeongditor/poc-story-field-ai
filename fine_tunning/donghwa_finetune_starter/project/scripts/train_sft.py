# scripts/train_sft.py
import os, argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # or "meta-llama/Meta-Llama-3.1-8B-Instruct"

def get_bnb_config(use_qlora: bool):
    if not use_qlora:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default=DEFAULT_MODEL)
    ap.add_argument('--train_file', type=str, default='data/processed/train.jsonl')
    ap.add_argument('--val_file', type=str, default='data/processed/val.jsonl')
    ap.add_argument('--output_dir', type=str, default='outputs/adapter')
    ap.add_argument('--use_qlora', action='store_true')
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--grad_accum', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--max_seq_len', type=int, default=2048)  # <- 그대로 인자로 받고 아래 토크나이즈에서만 사용
    args = ap.parse_args()

    bnb = get_bnb_config(args.use_qlora)

    print("Loading tokenizer/model...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype="auto",
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj","k_proj","v_proj","o_proj",
            "gate_proj","up_proj","down_proj",
        ],
    )

    ds_train = load_dataset('json', data_files=args.train_file, split='train')
    ds_val = load_dataset('json', data_files=args.val_file, split='train')

    # --- 중요: 텍스트 길이 안전 가드(토큰 기준 잘라내기) ---
    def format_example(example):
        messages = example["messages"]
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # 토큰 기준으로 잘라서 과도한 길이 방지
        ids = tok(text, add_special_tokens=False, truncation=True, max_length=args.max_seq_len)["input_ids"]
        trimmed = tok.decode(ids, skip_special_tokens=True)
        return {"text": trimmed}

    ds_train = ds_train.map(format_example, remove_columns=ds_train.column_names)
    ds_val = ds_val.map(format_example, remove_columns=ds_val.column_names)

    train_conf = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=20,
        save_steps=200,
        save_total_limit=3,
        bf16=True,
        # max_seq_length 파라미터 제거 (TRL 버전 호환)
        packing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_torch",
        report_to="none",
        eval_strategy="steps",
        eval_steps=200,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        peft_config=lora,
        args=train_conf,
        formatting_func=None,
        dataset_text_field="text",
    )

    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    print(f"Saved adapter to {args.output_dir}")

if __name__ == '__main__':
    main()
