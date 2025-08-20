# 📖 동화/이미지 생성 API (FastAPI)

이 프로젝트는 **텍스트 입력** 또는 **오디오 입력(STT)** 을 통해 동화를 생성하고, 각 페이지별 일러스트 이미지를 자동으로 생성합니다. 최종 결과는 `result/생성시간대/` 폴더에 저장되며, 클라이언트에는 `result.zip` 파일이 응답으로 반환됩니다.

---

## 설치 & 준비물

### 0) 필수 준비물
- Python 3.10 ~ 3.12 (권장: 3.10 또는 3.11)
- OpenAI API Key  
  `.env` 파일에 아래처럼 저장:
  ```env
  OPENAI_API_KEY=sk-본인API키
  ```
- (옵션) 오디오 입력을 사용할 경우 `ffmpeg` 설치 필요
  - Windows: [ffmpeg builds](https://www.gyan.dev/ffmpeg/builds/) 다운로드 후 PATH 등록
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt-get install ffmpeg`

---

### 1) 의존성 설치
```bash
python -m venv venv
source venv/bin/activate     # (Windows: venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 실행 방법

### A. 텍스트 입력 모드
```bash
uvicorn main:app --reload --port 8000
```
- Swagger 문서: [http://localhost:8000/docs](http://localhost:8000/docs)

### B. 오디오(STT) 입력 모드
```bash
uvicorn main_whisper:app --reload --port 8000
```
- Swagger 문서: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API 명세

### 1. 텍스트 입력 모드 (`main.py`)

#### Endpoint
`POST /process`

#### Request Body (JSON)
```json
{
  "character": "굳건이",
  "age": 26,
  "sex": "남자",
  "storyContent": "오늘 점심을 먹었어. 칼국수 집에 갔는데 국물이 진짜 시원하더라...",
  "keyword": {
    "atmosphere": "따뜻한",
    "drawingStyle": "수채화"
  }
}
```

#### Response
- Content-Type: `application/zip`
- 포함 파일:
  - `title.txt` → 동화 제목
  - `story.txt` → `"N페이지:내용"` 형식의 텍스트
  - `story.json` → 페이지별 객체 리스트
    ```json
    [
      {"pageNumber": 1, "content": "...", "filename": "page_1.png"},
      {"pageNumber": 2, "content": "...", "filename": "page_2.png"}
    ]
    ```
  - `title.png` → 표지 이미지
  - `gen_img/page_*.png` → 각 페이지 이미지

---

### 2. 오디오 입력 모드 (`main_whisper.py`)

#### Endpoint
`POST /process`

#### Request (Form-Data)
- `data`: JSON 문자열 (필수)
  ```json
  {
    "character": "굳건이",
    "age": 26,
    "sex": "남자",
    "keyword": {"atmosphere": "따뜻한", "drawingStyle": "수채화"}
  }
  ```
- `audio`: 업로드 파일 (필수, wav/mp3)

#### 처리 과정
1. `whisper.py`를 통해 오디오 → 텍스트(STT)
2. 텍스트를 `storyContent`로 사용
3. 이후 텍스트 입력 모드와 동일한 파이프라인 실행

#### Response
- 동일하게 `result.zip` 반환

---

## 출력 구조

```
result/20250101_123045/
  ├─ title.txt
  ├─ story.txt
  ├─ story.json
  ├─ title.png
  ├─ gen_img/page_1.png
  ├─ gen_img/page_2.png
  └─ result.zip
```

---

## 내부 동작 원리

1. **문어체 변환**: `gpt_casual2formal.py`  
   - 입력 storyContent → GPT로 문어체 변환
2. **동화 생성**: `gpt_story.py`  
   - 5~8페이지, 각 200~400자(3~6문장) 가이드, 대화/묘사/감정 포함
   - `make_title`로 제목 자동 생성  
3. **이미지 생성**: `dalle_image.py`  
   - DALL·E3 기반, 키워드 반영 (분위기/스타일), 텍스트 금지 프롬프트 포함  
4. **STT 변환**(옵션): `whisper.py`  
   - Whisper 모델로 mp3/wav → 텍스트 변환  
5. **결과 묶음**: `main.py` / `main_whisper.py`  
   - txt/json/png 파일 저장 후 zip 압축

---

## 입력/출력 스키마 요약(명세)

### 텍스트 입력 모드 Request(JSON)
- `character` (string, required)
- `age` (integer, required)
- `sex` (string, required)
- `storyContent` (string, required)
- `keyword` (object, optional)
  - `atmosphere` (string, optional)
  - `drawingStyle` (string, optional)

### STT 입력 모드 Request(Form-Data)
- `data` (text, required): 위 텍스트 입력 JSON을 문자열로 전달
- `audio` (file, required): wav/mp3 파일 (권장 최대 30~60MB)

### 공통 Response (application/zip)
- 헤더:
  - `Content-Type: application/zip`
  - `Content-Disposition: attachment; filename="result.zip"`
- 포함 파일:
  - `title.txt`, `story.txt`, `story.json`, `title.png`, `gen_img/page_*.png`

### story.json 스키마
```json
[
  {
    "pageNumber": 1,
    "content": "string (3~6문장, 200~400자 가이드)",
    "filename": "page_1.png"
  }
]
```

### 생성 제약/가이드
- 페이지 수: 5~8페이지
- 한 페이지: 3~6문장, 200~400자(풍부한 묘사/대화/감정/배경/생각 포함)
- 인코딩: UTF-8

---

## ⚠️ 오류 응답(예시)
- 400 Bad Request (유효하지 않은 입력)
  ```json
  {"detail": "Invalid JSON in 'data' field"}
  ```
- 401/429 OpenAI 오류: API Key/쿼터 문제
- 500 Server Error: 외부 API 실패 또는 내부 처리 오류

---

## GPU 사용 가이드(선택)
- Windows + NVIDIA GPU 환경에서 GPU를 사용하려면:
  1) NVIDIA 드라이버 설치 (nvidia-smi로 확인)
  2) CUDA 버전에 맞는 PyTorch 설치 (Python 3.10~3.12 권장)
     ```bash
     pip uninstall -y torch torchvision torchaudio
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
  3) Python에서 확인
     ```python
     import torch
     print(torch.cuda.is_available())  # True여야 GPU 사용 가능
     print(torch.cuda.get_device_name(0))
     ```
- GPU가 인식되지 않으면 CPU로 자동 동작합니다.

---

## Docker (선택)
- CPU 전용 간단 실행: `python:3.10-slim` 베이스 사용 가능
- GPU 사용 시: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` 베이스 + `--gpus all` 실행 필요
- 엔트리포인트 예: `uvicorn main_whisper:app --host 0.0.0.0 --port 8000`

---

## 문제 해결(FAQ)
- 401/429 등 OpenAI 에러: API Key 확인, 요금제/쿼터 확인
- 이미지 생성 400 에러(content policy): 프롬프트에서 민감한 금지어 제거 필요
- whisper mp3 변환 실패: ffmpeg 설치 필요
- GPU 미인식: Python 3.13 미지원, PyTorch CUDA wheel 재설치, 드라이버/CUDA 확인
- story.txt 형식이 어긋남: 모델 답변에 안내문이 섞일 수 있어 필터링 로직이 적용됨(본문만 저장)

---

## 프로젝트 구조
```
fastapi/
  ├─ main.py              # 텍스트 입력 모드 API
  ├─ main_whisper.py      # 오디오 입력 모드 API
  ├─ api/
  │   ├─ gpt_casual2formal.py  # 구어체 → 문어체 변환
  │   ├─ gpt_story.py          # 동화/제목 생성
  │   ├─ dalle_image.py        # 이미지 생성
  │   └─ whisper.py            # 음성 → 텍스트(STT)
  ├─ requirements.txt
  ├─ README.md (본 문서)
  └─ result/ (생성물 저장)
```
