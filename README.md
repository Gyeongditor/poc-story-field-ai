# poc-story-field-ai

이 프로젝트는 음성(STT), GPT 기반 텍스트 변환 및 동화 생성, DALL·E/Stable Diffusion 기반 이미지 생성을 통합적으로 다루는 AI 파이프라인 예제입니다.

## 주요 폴더 구조 및 역할

- integration_test: 전체 파이프라인 통합 테스트 및 예제 실행 코드
- STT: 음성 인식(STT) 관련 코드 및 실험 노트북
  - whisper: Whisper 기반 음성 인식 실험
  - poc_kospeech: KoSpeech 기반 음성 인식 실험
  - kospeech: KoSpeech 관련 실험 노트북
- STORY: 텍스트 생성 및 동화 스토리라인 생성 관련 코드/노트북
  - StoryLine: 스토리라인 생성 노트북
  - LLM: 대형 언어 모델(LLM) 기반 동화 생성 실험
  - GPT_API: GPT API 활용 텍스트 생성 실험
- IMAGE: DALL·E, Stable Diffusion 등 이미지 생성 관련 노트북 및 결과물

## 주요 기능 요약

1. **음성(STT) → 텍스트 변환**
2. **구어체 → 문어체 변환(GPT)**
3. **동화 스토리/페이지별 텍스트 생성(GPT)**
4. **페이지별 이미지 생성(DALL·E, Stable Diffusion)**

## 실행 환경 및 의존성
- Python 3.8 이상 권장
- requirements.txt, windows_requirements.txt 참고

---

각 폴더별 README.md에서 세부 파일 설명을 확인할 수 있습니다.