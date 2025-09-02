#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


def load_model(base_model: str, adapter_dir: str, bf16: bool = True, quantize_4bit: bool = True):
    # Gracefully disable 4bit when bitsandbytes is not available (e.g., Python 3.12)
    if quantize_4bit:
        try:
            import bitsandbytes as _bnb  # noqa: F401
            _bnb_available = True
        except Exception:
            _bnb_available = False
        if not _bnb_available:
            print("[WARN] bitsandbytes가 감지되지 않아 4bit 양자화를 비활성화합니다 (--quantize_4bit false 적용).")
            quantize_4bit = False

    load_kwargs = {"device_map": "auto"}
    if quantize_4bit:
        quant_dtype = torch.bfloat16 if bf16 else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=quant_dtype,
        )
        load_kwargs["quantization_config"] = quant_config
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    model = PeftModel.from_pretrained(model, adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def build_prompt(instruction: str, user_input: str) -> str:
    return (
        f"### 지시사항:\n{instruction}\n\n"
        f"### 입력:\n{user_input}\n\n"
        f"### 응답:\n"
    )


def generate(
    model,
    tokenizer,
    instruction: str,
    user_input: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    prompt = build_prompt(instruction, user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Strip the prompt
    return output_text.split("### 응답:")[-1].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="beomi/llama-2-ko-7b")
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--instruction", type=str, default="다음 동화의 페이지 본문을 자연스러운 한국어로 작성하세요.")
    # 자유 입력 또는 구조화 입력 중 하나 사용
    parser.add_argument("--input", dest="user_input", type=str, required=False, help="직접 구성한 입력 문자열")

    # 구조화 입력 필드 (fastapi 스키마 기반)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--character", type=str, default=None)
    parser.add_argument("--age", type=int, default=None)
    parser.add_argument("--sex", type=str, default=None)
    parser.add_argument("--atmosphere", type=str, default=None)
    parser.add_argument("--drawingStyle", type=str, default=None)
    parser.add_argument("--storyContent", type=str, default=None)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--sentences", type=int, default=3)
    parser.add_argument("--words", type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_sample", action="store_true")
    parser.add_argument("--bf16", type=lambda s: s.lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--quantize_4bit", type=lambda s: s.lower() in {"1", "true", "yes"}, default=True)
    args = parser.parse_args()

    # 필요 시 구조화 입력으로 user_input 생성
    if not args.user_input:
        # 제목 결정
        if args.title:
            title = args.title
        elif args.character:
            title = f"{args.character}가 등장하는 이야기"
        else:
            title = "동화 이야기"

        # 분류(분위기/그림체) 결합
        cls_parts = []
        if args.atmosphere:
            cls_parts.append(args.atmosphere)
        if args.drawingStyle:
            cls_parts.append(args.drawingStyle)
        classification = ", ".join(cls_parts) if cls_parts else "-"

        # 등장인물 문자열
        if args.character and args.sex:
            character_line = f"등장인물: {args.character} ({args.sex})"
        elif args.character:
            character_line = f"등장인물: {args.character}"
        else:
            character_line = None

        lines = [
            f"제목: {title}",
            f"분류: {classification}",
            f"읽기 연령: {args.age if args.age is not None else '-'}",
            f"페이지 번호: {args.page}",
            f"문장 수: {args.sentences}",
            f"단어 수: {args.words}",
        ]
        if character_line:
            lines.append(character_line)
        if args.storyContent:
            lines.append(f"핵심내용: {args.storyContent}")

        args.user_input = "\n".join(lines)

    model, tokenizer = load_model(args.base_model, args.adapter_dir, bf16=args.bf16, quantize_4bit=args.quantize_4bit)
    text = generate(
        model,
        tokenizer,
        instruction=args.instruction,
        user_input=args.user_input,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
    )
    print(text)


if __name__ == "__main__":
    main()


