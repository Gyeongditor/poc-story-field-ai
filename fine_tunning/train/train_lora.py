#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoRA Supervised Fine-Tuning (SFT) script for instruction-style JSONL datasets.

Expected dataset schema (one JSON per line):
{
  "instruction": str,
  "input": str,
  "output": str
}

This script formats each sample as:

### 지시사항:
{instruction}

### 입력:
{input}

### 응답:
{output}

Labels are masked so that loss is only computed on the "응답" part (via
DataCollatorForCompletionOnlyLM with response_template="### 응답:").

Works with 4-bit QLoRA by default on a single GPU.
"""

import os
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Callable

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer
from peft import LoraConfig, TaskType


def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y", "t"}


def build_formatting_func(response_header: str) -> Callable[[Dict[str, Any]], List[str]]:
    def _format(example: Dict[str, Any]) -> List[str]:
        instruction = example.get("instruction") or ""
        user_input = example.get("input") or ""
        output = example.get("output") or ""
        text = (
            f"### 지시사항:\n{instruction}\n\n"
            f"### 입력:\n{user_input}\n\n"
            f"{response_header}\n{output}"
        )
        return [text]

    return _format


@dataclass
class ScriptArgs:
    model_name_or_path: str
    data_dir: str
    train_file: str
    eval_file: str
    output_dir: str
    max_seq_len: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    learning_rate: float
    logging_steps: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    warmup_ratio: float
    weight_decay: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: str
    seed: int
    bf16: bool
    fp16: bool
    quantize_4bit: bool
    use_gradient_checkpointing: bool
    response_template: str


def parse_args() -> ScriptArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="beomi/llama-2-ko-7b")
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "story"))
    parser.add_argument("--train_file", type=str, default="hf_instruction_train.jsonl")
    parser.add_argument("--eval_file", type=str, default="hf_instruction_val.jsonl")
    parser.add_argument("--output_dir", type=str, default="./lora-out")

    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
        help="Comma-separated module names for LoRA",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--fp16", type=str2bool, default=False)
    parser.add_argument("--quantize_4bit", type=str2bool, default=True)
    parser.add_argument("--use_gradient_checkpointing", type=str2bool, default=True)
    parser.add_argument(
        "--response_template",
        type=str,
        default="### 응답:",
        help="Prefix string that marks the start of the response segment.",
    )

    args = parser.parse_args()

    return ScriptArgs(
        model_name_or_path=args.model_name_or_path,
        data_dir=args.data_dir,
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        quantize_4bit=args.quantize_4bit,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        response_template=args.response_template,
    )


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    data_files = {
        "train": os.path.join(args.data_dir, args.train_file),
        "validation": os.path.join(args.data_dir, args.eval_file),
    }
    dataset = load_dataset("json", data_files=data_files)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Model loading (4-bit QLoRA by default)
    load_kwargs: Dict[str, Any] = {"device_map": "auto"}
    if args.quantize_4bit:
        load_kwargs.update(
            dict(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            )
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)

    # Enable gradient checkpointing for memory efficiency
    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # LoRA config
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        inference_mode=False,
    )

    # Trainer args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=args.fp16 and torch.cuda.is_available(),
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type="cosine",
        report_to=["none"],
        seed=args.seed,
        dataloader_num_workers=2,
    )

    # Build formatting function and data collator for label masking
    formatting_func = build_formatting_func(args.response_template)

    # When using TRL SFTTrainer with text formatting, we can either use
    # DataCollatorForLanguageModeling (train_on_prompt=True/False) or
    # Completion-only collator. Here we rely on SFTTrainer's internal masking via
    # response_template parameter.

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        formatting_func=formatting_func,
        max_seq_length=args.max_seq_len,
        packing=False,
        dataset_num_proc=2,
        response_template=args.response_template,
        train_on_prompt=False,
    )

    last_ckpt = None
    if os.path.isdir(args.output_dir):
        last_ckpt = get_last_checkpoint(args.output_dir)

    trainer.train(resume_from_checkpoint=last_ckpt)

    # Save adapter only
    trainer.save_model()  # saves the PEFT adapter into output_dir
    tokenizer.save_pretrained(args.output_dir)

    print("[DONE] LoRA SFT training complete. Adapters saved at:", args.output_dir)


if __name__ == "__main__":
    main()


