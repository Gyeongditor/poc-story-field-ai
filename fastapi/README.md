# FastAPI 동화 생성/이미지 API

## 주요 기능
- 자연어로 입력한 storyContent(일상/키워드 등)로 동화책 텍스트와 이미지를 자동 생성
- OpenAI GPT, DALL-E API 활용 (화풍 일관성, 텍스트 없는 이미지)
- 결과물: story.txt(페이지별 동화), title.txt(제목), story.json, 표지/페이지별 이미지, result.zip
- 모든 결과는 result/{생성시간대}/ 폴더에 저장

## 설치 및 실행

1. 패키지 설치
   ```bash
   pip install -r requirements.txt
   ```

2. 환경변수 설정 (OpenAI API 키)
   - .env 파일에 아래와 같이 작성 (fastapi 폴더에 위치)
     ```
     OPENAI_API_KEY=sk-여기에_본인_API키_입력
     ```
   - 또는 터미널에서 직접 설정

3. 서버 실행
   - **텍스트 입력 기반 동화 생성**
     ```bash
     uvicorn main:app --reload
     ```
   - **오디오(STT) 기반 동화 생성**
     ```bash
     uvicorn main_whisper:app --reload
     ```

4. 테스트
   - Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Postman, Python 등으로 multipart/form-data 응답 확인

## 입력 예시

### main.py (텍스트 입력)
```json
{
  "character": "굳건이",
  "age": 26,
  "sex": "남자",
  "storyContent": "오늘 점심을 먹었어. 칼국수 집에 갔는데 국물이 진짜 시원하더라. 같이 간 친구가 김치전을 시켜서 한 입 먹었는데 그것도 맛있었어. 배부르게 먹고 나니까 갑자기 낮잠이 너무 오더라. 그래서 근처 카페 가서 아이스 아메리카노 한 잔 시켜놓고 좀 쉬었어. 아 맞다, 카페에서 우연히 초등학교 동창을 만나서 깜짝 놀랐지. 오랜만에 만나서 잠깐 얘기 나누고 연락처도 교환했어. 오늘은 생각보다 특별한 하루였네.",
  "keyword": {
    "atmosphere": "따뜻한",
    "drawingStyle": "수채화"
  }
}
```

### main_whisper.py (오디오 입력)
- **form-data**로 전송
  - key: data, value: (아래 JSON을 string으로 입력, type: Text)
    ```json
    {
      "character": "굳건이",
      "age": 26,
      "sex": "남자",
      "keyword": {
        "atmosphere": "따뜻한",
        "drawingStyle": "수채화"
      }
    }
    ```
  - key: audio, value: (wav/mp3 파일 업로드, type: File)

## 출력 예시
- story.txt: 각 페이지별 동화 내용 (예: 1페이지:내용)
- title.txt: 동화 제목
- story.json: 페이지별 동화+이미지 파일명 리스트
- thumbnail: 표지 이미지 (title.png)
- page_image_1, page_image_2, ...: 각 페이지별 이미지
- result.zip: 전체 결과물 압축본

## 주요 구현/프롬프트 특징
- storyContent의 실제 내용을 동화에 반드시 반영하도록 프롬프트에서 강하게 요구
- 동화는 5~8페이지, 각 페이지는 'N페이지:내용' 한 줄로만 생성
- 이미지 프롬프트에 일관된 화풍, 텍스트/워터마크/자막/서명 등 금지 문구 포함
- 모든 결과물은 result/{생성시간대}/ 폴더에 저장됨

## 참고/주의사항
- OpenAI API 사용량에 따라 요금이 발생할 수 있음
- DALL-E 프롬프트에 민감한 금지어(언어/인종/국적 등)는 사용하지 않음
- story.txt는 안내문/코드/마크다운/빈 줄 없이 실제 동화 본문만 저장
- 여러 번 실행 시 result/ 폴더에 타임스탬프별로 결과가 쌓임
- main_whisper.py는 오디오(STT) 기반 동화 생성 전용 엔드포인트임