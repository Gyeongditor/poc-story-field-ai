# 동화 생성 파이프라인 (STT → 동화책)

이 프로젝트는 **음성(STT)** 으로부터 자동으로 동화를 생성하는 파이프라인입니다.  
구성 요소:
1. **STT-preprocess.py** : Whisper 전사문 → 정제(clean text)
2. **gen_story.py** : clean text → 16p 동화 아웃라인(JSON)
3. **gen_img.py** : 아웃라인 → 삽화 프롬프트(JSON)

## 📂 파일 구조
