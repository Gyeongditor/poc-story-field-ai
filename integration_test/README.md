# Integration Test

이 폴더는 Whisper(STT), GPT API(문어체 변환/동화 생성), DALL·E(이미지 생성) 기능의 통합 테스트를 위한 코드와 결과물을 관리합니다.

## 파일 구성 및 역할
- **main.py**: 전체 파이프라인 실행 (음성→문어체 변환→동화 생성→페이지별 이미지 생성)
- **stt_whisper.py**: Whisper로 음성(STT) 인식 (transcribe 함수)
- **gpt_casual2formal.py**: GPT API로 구어체→문어체 변환 (casual_to_formal 함수)
- **gpt_story.py**: GPT API로 동화 생성 (generate_story 함수)
- **dalle_image.py**: DALL·E로 페이지별 이미지 생성 (generate_images 함수, gen_img 폴더에 저장)
- **travel_test.wav**: 샘플 음성 파일
- **.env**: OpenAI API 키 등 환경 변수 파일 (예시: OPENAI_API_KEY=sk-...)

## 전체 파이프라인 흐름
1. **음성 인식**: travel_test.wav → Whisper로 텍스트 변환
2. **문어체 변환**: 구어체 텍스트 → GPT로 문어체 변환
3. **동화 생성**: 문어체 텍스트 → GPT로 5페이지 동화 생성 ([PAGE] 구분)
4. **이미지 생성**: 각 페이지별 프롬프트로 DALL·E 이미지 생성 (gen_img/page_1.png ...)

## 실행 방법
1. `.env` 파일에 OpenAI API 키 입력 (OPENAI_API_KEY=sk-...)
2. 필요한 패키지 설치 (requirements.txt 참고)
3. main.py 실행

## 결과물
- 생성된 동화: main.py 출력
- 생성된 이미지: gen_img/page_1.png, page_2.png, ...
