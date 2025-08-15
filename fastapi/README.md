# FastAPI 동화 생성 서버

## 폴더 구조 및 주요 파일 설명

- **server.py**: FastAPI 기반 API 서버. 구어체 텍스트를 받아 동화(문어체 변환, 동화 생성, 페이지별 이미지, 표지 이미지, 제목) 생성 및 결과 반환.
- **dalle_image.py**: DALL·E API를 이용해 동화 각 페이지 및 표지 이미지를 생성하는 함수 제공.
- **gpt_story.py**: OpenAI GPT API로 동화 본문 생성, 파일 저장 함수 제공.
- **gpt_casual2formal.py**: 구어체를 문어체로 변환하는 GPT API 함수 제공.
- **travel_test_whisper.txt**: 샘플 STT 결과 텍스트 파일(실제 서비스와 직접적 연동 없음).


## 전체 동작 흐름
1. **POST /process** 엔드포인트로 구어체 텍스트(stt)와 키워드(optional)를 전달
2. 구어체 → 문어체 변환
3. 문어체 텍스트로 동화 본문 생성 (5페이지, [PAGE] 구분)
4. 동화 제목 생성 및 저장 (result/title.txt)
5. 동화 표지 이미지 생성 및 저장 (result/title.png)
6. 각 페이지별 이미지 생성 및 저장 (result/gen_img/page_1.png ...)
7. 동화 본문 전체 저장 (result/gen_story/story.txt)
8. 결과 JSON 반환 (제목, 표지 이미지 경로, 각 페이지, 각 이미지 경로)


## 도커로 실행하는 방법

1. **이미지 빌드**
   ```sh
   docker build -t storyfield-fastapi .
   ```
2. **실행**
   ```sh
   docker run --env-file .env -p 8000:8000 -v C:/Users/chan/poc-story-field-ai/fastapi/result:/app/result storyfield-fastapi
   ```
   - .env 파일에 OPENAI_API_KEY=sk-... 포함 필요
   - result 폴더는 호스트와 공유됨

3. **API 호출 예시**
   ```sh
   curl -X POST "http://localhost:8000/process" \
        -H "Content-Type: application/json" \
        -d '{"stt": "옛날 옛적에...", "keyword": ["여행", "마법"]}'
   ```


## 각 파일별 상세 설명

- **server.py**
  - FastAPI 서버 엔트리포인트
  - /process POST: stt(구어체)와 keyword(optional) 입력 → 동화, 제목, 표지, 이미지 생성
- **dalle_image.py**
  - generate_images: 각 페이지별 이미지 생성
  - generate_title_image: 표지 이미지(title.png) 생성
- **gpt_story.py**
  - generate_story: 프롬프트로 동화 본문/제목 생성
  - save_story: 동화 본문 파일 저장
- **gpt_casual2formal.py**
  - casual_to_formal: 구어체 → 문어체 변환 함수


## 기타
- FastAPI 서버는 8000번 포트에서 실행됩니다.
- 결과물은 result/ 하위에 저장됩니다.
- .env 파일은 반드시 이미지에 포함하지 말고, 실행 시 --env-file로 주입하세요.
http://localhost:8000/docs 환경에서 테스트 해보세요
예시: {
  "stt": travel_test_whisper에 있는 내용,
  "keyword": ["따뜻한 색감", "주인공: 민수", "성별: 남자", "분위기: 모험"]
}