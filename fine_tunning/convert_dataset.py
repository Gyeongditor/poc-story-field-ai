#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple


def read_json(file_path: Path) -> Any:
    with file_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Tuple[str, str]]) -> None:
    with path.open('w', encoding='utf-8') as f:
        f.write('image_path,caption\n')
        for image_path, caption in rows:
            # escape quotes and commas in caption
            safe_caption = caption.replace('"', '""')
            f.write(f'"{image_path}","{safe_caption}"\n')


def collect_story_files(root: Path) -> Dict[str, List[Path]]:
    story_root = root / '015.동화 줄거리 생성 데이터' / '3.개방데이터' / '1.데이터'
    train_dir = story_root / 'Training' / '01.원천데이터'
    val_dir = story_root / 'Validation' / '01.원천데이터'

    train_files: List[Path] = []
    val_files: List[Path] = []

    if train_dir.exists():
        train_files = sorted([p for p in train_dir.rglob('*.json')])
    if val_dir.exists():
        val_files = sorted([p for p in val_dir.rglob('*.json')])

    return {
        'train': train_files,
        'val': val_files,
    }


# Caption 데이터 수집 함수 제거 - 줄거리 데이터만 사용


def make_story_samples(files: List[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    hf_instruction_rows: List[Dict[str, Any]] = []
    openai_chat_rows: List[Dict[str, Any]] = []
    hf_causal_rows: List[Dict[str, Any]] = []

    for fp in files:
        try:
            data = read_json(fp)
        except Exception:
            continue

        title = data.get('title') or ''
        classification = data.get('classification') or ''
        read_age = data.get('readAge') or ''
        paragraph_info: List[Dict[str, Any]] = data.get('paragraphInfo') or []

        # Instruction/chat: one sample per paragraph
        for p in paragraph_info:
            out_text = p.get('srcText') or ''
            if not out_text:
                continue
            page_num = p.get('srcPage')
            sent_ea = p.get('srcSentenceEA')
            word_ea = p.get('srcWordEA')

            instruction = '다음 동화의 페이지 본문을 자연스러운 한국어로 작성하세요.'
            user_input = (
                f'제목: {title}\n'
                f'분류: {classification}\n'
                f'읽기 연령: {read_age}\n'
                f'페이지 번호: {page_num}\n'
                f'문장 수: {sent_ea}\n'
                f'단어 수: {word_ea}'
            )
            hf_instruction_rows.append({
                'instruction': instruction,
                'input': user_input,
                'output': out_text,
                'meta': {
                    'source_file': str(fp)
                }
            })

            openai_chat_rows.append({
                'messages': [
                    {'role': 'system', 'content': '당신은 한국어 동화 문장을 쓰는 작가입니다.'},
                    {'role': 'user', 'content': instruction + "\n\n" + user_input},
                    {'role': 'assistant', 'content': out_text},
                ],
                'meta': {
                    'source_file': str(fp)
                }
            })

        # Causal LM: one sample per book (concatenate in ascending page order if possible)
        try:
            paragraphs_sorted = sorted(
                [p for p in paragraph_info if p.get('srcText')],
                key=lambda x: (x.get('srcPage') is None, x.get('srcPage'))
            )
        except Exception:
            paragraphs_sorted = [p for p in paragraph_info if p.get('srcText')]

        full_text_parts: List[str] = []
        header = [
            f'[제목] {title}',
            f'[분류] {classification}',
            f'[연령] {read_age}',
        ]
        full_text_parts.append('\n'.join(header))
        for p in paragraphs_sorted:
            pg = p.get('srcPage')
            txt = p.get('srcText') or ''
            if not txt:
                continue
            if pg is not None:
                full_text_parts.append(f'[페이지 {pg}] {txt}')
            else:
                full_text_parts.append(txt)
        full_text = '\n'.join(full_text_parts)
        if full_text.strip():
            hf_causal_rows.append({'text': full_text, 'meta': {'source_file': str(fp)}})

    return hf_instruction_rows, openai_chat_rows, hf_causal_rows


# Caption 샘플 생성 함수 제거 - 줄거리 데이터만 사용


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.', help='데이터 루트 경로 (기본: 현재 경로)')
    parser.add_argument('--out', type=str, default='exports', help='결과 저장 폴더')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    project_root = Path(args.root).resolve()
    out_root = (project_root / args.out).resolve()
    ensure_dir(out_root)

    random.seed(args.seed)

    # 1) Story dataset
    story_files = collect_story_files(project_root)
    story_train_files = story_files.get('train', [])
    story_val_files = story_files.get('val', [])

    story_out = out_root / 'story'
    ensure_dir(story_out)

    # Train
    hf_inst_train, openai_chat_train, hf_causal_train = make_story_samples(story_train_files)
    # Val
    hf_inst_val, openai_chat_val, hf_causal_val = make_story_samples(story_val_files)

    write_jsonl(story_out / 'hf_instruction_train.jsonl', hf_inst_train)
    write_jsonl(story_out / 'hf_instruction_val.jsonl', hf_inst_val)
    write_jsonl(story_out / 'openai_chat_train.jsonl', openai_chat_train)
    write_jsonl(story_out / 'openai_chat_val.jsonl', openai_chat_val)
    write_jsonl(story_out / 'hf_causal_train.jsonl', hf_causal_train)
    write_jsonl(story_out / 'hf_causal_val.jsonl', hf_causal_val)

    # Caption 데이터는 사용하지 않음 - 줄거리 데이터만 사용

    # 3) Simple README with usage tips
    readme = out_root / 'README.md'
    readme.write_text(
        (
            '# 동화 줄거리 생성 파인튜닝 데이터\n\n'
            '이 폴더에는 동화 줄거리 생성 데이터만으로 구축된 HuggingFace 및 OpenAI GPT 파인튜닝용 데이터가 포함되어 있습니다.\n\n'
            '## 데이터 구조\n'
            '- story/: 동화 줄거리(텍스트) 기반 데이터\n'
            '  - hf_instruction_*.jsonl: instruction-tuning (Alpaca 스타일: instruction/input/output)\n'
            '  - openai_chat_*.jsonl: OpenAI 채팅 파인튜닝 포맷(messages 배열)\n'
            '  - hf_causal_*.jsonl: causal LM 포맷({"text"})\n\n'
            '## 데이터 개수\n'
            '- Training: 1,603개 동화 파일 → 각 포맷별 샘플 수 생성\n'
            '- Validation: 216개 동화 파일 → 각 포맷별 샘플 수 생성\n'
            '- 각 파일의 paragraph 수만큼 instruction 샘플 생성\n'
            '- 각 파일당 1개의 causal LM 샘플 생성\n\n'
            '## HuggingFace 예시\n'
            'Python에서 datasets 로드 예시:\n\n'
            '```python\n'
            'from datasets import load_dataset\n'
            'ds = load_dataset("json", data_files={"train": "story/hf_instruction_train.jsonl", "validation": "story/hf_instruction_val.jsonl"})\n'
            'print(ds)\n'
            '```\n\n'
            'Causal LM 예시(Transformers Trainer):\n\n'
            '```python\n'
            'from datasets import load_dataset\n'
            'from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer\n\n'
            'ds = load_dataset("json", data_files={"train": "story/hf_causal_train.jsonl", "validation": "story/hf_causal_val.jsonl"})\n'
            'tok = AutoTokenizer.from_pretrained("gpt2")\n'
            'def tok_fn(batch):\n'
            '    return tok(batch["text"], truncation=True)\n'
            'ds_tok = ds.map(tok_fn, batched=True, remove_columns=ds["train"].column_names)\n'
            'model = AutoModelForCausalLM.from_pretrained("gpt2")\n'
            'args = TrainingArguments(output_dir="./out", per_device_train_batch_size=2, num_train_epochs=1)\n'
            'trainer = Trainer(model=model, args=args, train_dataset=ds_tok["train"], eval_dataset=ds_tok["validation"])\n'
            'trainer.train()\n'
            '```\n\n'
            '## OpenAI 파인튜닝 예시\n'
            '- story/openai_chat_train.jsonl 파일을 업로드하여 사용하세요.\n'
            '- OpenAI CLI 예시 (모델 이름과 API 사용법은 최신 문서를 참고):\n\n'
            '```bash\n'
            'openai files upload --purpose fine-tune --file story/openai_chat_train.jsonl\n'
            'openai files upload --purpose fine-tune --file story/openai_chat_val.jsonl\n'
            '# 이후 fine-tuning job 생성\n'
            '# openai fine_tuning.jobs.create -t <TRAIN_FILE_ID> -v <VAL_FILE_ID> -m gpt-4o-mini\n'
            '```\n\n'
            '## 특징\n'
            '- 각 동화의 메타데이터(제목, 분류, 연령)를 포함한 풍부한 컨텍스트\n'
            '- 페이지별 세부 정보와 함께한 구조화된 학습 데이터\n'
            '- 한국어 동화 생성에 특화된 고품질 데이터셋\n'
        ),
        encoding='utf-8'
    )

    # 4) 간단한 통계 출력
    print('[DONE] 줄거리 생성 데이터 변환 완료')
    print('Story (instruction) - train:', len(hf_inst_train), 'val:', len(hf_inst_val))
    print('Story (openai chat) - train:', len(openai_chat_train), 'val:', len(openai_chat_val))
    print('Story (causal) - train:', len(hf_causal_train), 'val:', len(hf_causal_val))
    print('원본 데이터 - train: 1603개 파일, val: 216개 파일')


if __name__ == '__main__':
    main()


