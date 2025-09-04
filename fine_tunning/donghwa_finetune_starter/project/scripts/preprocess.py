
import argparse, json, os, re, io, zipfile
from pathlib import Path
import pandas as pd

TARGET_PAGES = 5
SENTS_PER_PAGE = (3, 4)  # (min, max)

KEY_MAP = {
    "character": ["character", "캐릭터", "주인공", "char", "person"],
    "age": ["age", "나이"],
    "sex": ["sex", "gender", "성별"],
    "story": ["story", "storyContent", "text", "본문", "내용", "story_text"],
    "atmosphere": ["atmosphere", "분위기"],
    "drawingStyle": ["drawingStyle", "style", "화풍", "그림체"],
    "title": ["title", "제목"],
    "pages": ["pages", "storyPages", "쪽", "page_list"],
}

END_TOKENS = r"[.!?。？！]|다\.|요\.|요\?|다\?"
SENT_SPLIT_REGEX = re.compile(f"({END_TOKENS})\s+")

def sentence_tokenize(text: str):
    if not text:
        return []
    parts = SENT_SPLIT_REGEX.split(text.strip())
    # parts 예: [문장조각, 구분자, 문장조각, 구분자, ... , 마지막조각(옵션)]
    sents = []
    i = 0
    while i < len(parts):
        chunk = parts[i].strip() if parts[i] else ""
        sep = parts[i+1] if i+1 < len(parts) else ""
        if chunk:
            sents.append((chunk + (sep or "")).strip())
        i += 2
    return [s for s in sents if s]

def first_key(d, candidates):
    for k in candidates:
        if k in d and d[k] is not None and d[k] != "":
            return k
    return None

def normalize_record(rec: dict):
    out = {
        "character": rec.get(first_key(rec, KEY_MAP["character"]), None),
        "age": rec.get(first_key(rec, KEY_MAP["age"]), None),
        "sex": rec.get(first_key(rec, KEY_MAP["sex"]), None),
        "title": rec.get(first_key(rec, KEY_MAP["title"]), None),
        "keyword": {
            "atmosphere": rec.get(first_key(rec, KEY_MAP["atmosphere"]), None),
            "drawingStyle": rec.get(first_key(rec, KEY_MAP["drawingStyle"]), None),
        },
    }

    pages_key = first_key(rec, KEY_MAP["pages"]) if KEY_MAP.get("pages") else None
    if pages_key and isinstance(rec.get(pages_key), (list, tuple)) and len(rec[pages_key]) >= 1:
        out["pages"] = [str(x).strip() for x in rec[pages_key] if str(x).strip()]
    else:
        story_key = first_key(rec, KEY_MAP["story"]) if KEY_MAP.get("story") else None
        out["story"] = str(rec.get(story_key, "")).strip() if story_key else None

    return out

def split_to_pages(text: str, pages=TARGET_PAGES):
    if not text:
        return []
    sents = sentence_tokenize(text)
    if not sents:
        return []
    target_min, target_max = SENTS_PER_PAGE
    target_per_page = max(target_min, min(target_max, max(3, len(sents)//pages)))

    pages_list = []
    i = 0
    for p in range(pages):
        chunk = sents[i:i+target_per_page]
        if not chunk:
            break
        pages_list.append(" ".join(chunk))
        i += target_per_page

    rest = sents[i:]
    j = 0
    while rest and j < len(pages_list):
        pages_list[j] += (" " + rest.pop(0))
        j = (j + 1) % len(pages_list)

    while len(pages_list) < pages:
        pages_list.append("")
    return pages_list[:pages]

def make_user_json(nrm: dict):
    payload = {
        "character": nrm.get("character") or "토끼",
        "age": nrm.get("age") or 5,
        "sex": nrm.get("sex") or "여",
        "storyContent": nrm.get("story") or "",
        "keyword": {
            "atmosphere": (nrm.get("keyword") or {}).get("atmosphere") or "따뜻한",
            "drawingStyle": (nrm.get("keyword") or {}).get("drawingStyle") or "수채화",
        },
    }
    return json.dumps(payload, ensure_ascii=False)

def format_pages(pages):
    out = []
    for idx, p in enumerate(pages, 1):
        p = (p or "").strip()
        out.append(f"### {idx}쪽\n" + (p if p else "(비워둠)"))
    return "\n\n".join(out)

SYSTEM_PROMPT = (
    "너는 유아동 동화 작가야. 사용자의 JSON 입력을 읽고, 한국어로 5쪽 동화를 작성해. "
    "각 쪽은 3~4문장, 온화한 어조(요/다체 혼용 가능). 제목은 생략하고 본문만 출력. "
    "페이지 표시는 '### 1쪽'처럼 달고, 내용 외 불필요한 설명은 쓰지 마."
)

def record_to_messages(nrm: dict):
    if "pages" in nrm and nrm["pages"]:
        pages = nrm["pages"]
        if len(pages) < TARGET_PAGES:
            pages = (pages + [""] * TARGET_PAGES)[:TARGET_PAGES]
    else:
        pages = split_to_pages(nrm.get("story", ""), TARGET_PAGES)
        if len(pages) < TARGET_PAGES:
            pages += [""] * (TARGET_PAGES - len(pages))
    assistant_text = format_pages(pages)

    user_json = make_user_json(nrm)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_json},
            {"role": "assistant", "content": assistant_text},
        ]
    }

def read_any(path: Path):
    recs = []
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, 'r') as z:
            for name in z.namelist():
                if name.endswith('/'):
                    continue
                low = name.lower()
                with z.open(name) as f:
                    if low.endswith('.jsonl'):
                        for line in io.TextIOWrapper(f, encoding='utf-8'):
                            if line.strip():
                                recs.append(json.loads(line))
                    elif low.endswith('.json'):
                        text = io.TextIOWrapper(f, encoding='utf-8').read()
                        data = json.loads(text)
                        if isinstance(data, list):
                            recs.extend(data)
                        elif isinstance(data, dict):
                            recs.append(data)
                    elif low.endswith('.csv'):
                        df = pd.read_csv(f)
                        recs.extend(df.to_dict(orient='records'))
                    elif low.endswith('.xlsx'):
                        df = pd.read_excel(f)
                        recs.extend(df.to_dict(orient='records'))
    elif path.is_dir():
        for p in path.rglob('*'):
            if p.is_dir():
                continue
            low = p.name.lower()
            if low.endswith('.jsonl'):
                with open(p, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            recs.append(json.loads(line))
            elif low.endswith('.json'):
                data = json.load(open(p, 'r', encoding='utf-8'))
                if isinstance(data, list):
                    recs.extend(data)
                else:
                    recs.append(data)
            elif low.endswith('.csv'):
                df = pd.read_csv(p)
                recs.extend(df.to_dict(orient='records'))
            elif low.endswith('.xlsx'):
                df = pd.read_excel(p)
                recs.extend(df.to_dict(orient='records'))
    else:
        low = path.name.lower()
        if low.endswith('.jsonl'):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        recs.append(json.loads(line))
        elif low.endswith('.json'):
            data = json.load(open(path, 'r', encoding='utf-8'))
            if isinstance(data, list):
                recs.extend(data)
            else:
                recs.append(data)
        elif low.endswith('.csv'):
            df = pd.read_csv(path)
            recs.extend(df.to_dict(orient='records'))
        elif low.endswith('.xlsx'):
            df = pd.read_excel(path)
            recs.extend(df.to_dict(orient='records'))
    return recs

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', type=str, required=True, help='원본 .zip 또는 폴더/파일 경로')
    ap.add_argument('--out_dir', type=str, default='data/processed')
    ap.add_argument('--val_ratio', type=float, default=0.05)
    args = ap.parse_args()

    in_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = read_any(in_path)
    print(f"Loaded {len(raw)} raw records")

    items = []
    for r in raw:
        nrm = normalize_record(r)
        msg = record_to_messages(nrm)
        items.append(msg)

    n = len(items)
    v = max(1, int(n * args.val_ratio)) if n > 1 else 1
    val = items[:v]
    train = items[v:]

    with open(out_dir / 'train.jsonl', 'w', encoding='utf-8') as f:
        for obj in train:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with open(out_dir / 'val.jsonl', 'w', encoding='utf-8') as f:
        for obj in val:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved train={len(train)}, val={len(val)} to {out_dir}")

if __name__ == '__main__':
    main()
