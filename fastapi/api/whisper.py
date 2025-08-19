import whisper
import torch
import os

# 전역 모델 캐시 (모델 크기별로 1회만 로드)
_model_cache = {}

def get_device(): # GPU 사용 or CPU 사용
    """GPU가 있으면 cuda, 없으면 cpu 반환"""
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_whisper_model(model_size="medium"): # whisper 모델 캐싱
    """모델 크기별로 whisper 모델을 캐싱해서 반환"""
    global _model_cache
    if model_size not in _model_cache:
        device = get_device()
        _model_cache[model_size] = whisper.load_model(model_size, device=device)
    return _model_cache[model_size]

# 오디오 파일을 받아 텍스트로 변환하는 함수 (main.py에서 import해서 사용)
def transcribe_audio(audio_path, model_size="medium"):
    """
    audio_path: 오디오 파일 경로 (wav, mp3 등)
    model_size: whisper 모델 크기 (base, small, medium 등)
    반환값: 인식된 전체 텍스트(str)
    """
    model = load_whisper_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"]

# # 사용 예시 (직접 실행 시)
# if __name__ == "__main__":
#     # 예시 오디오 파일 경로
#     audio_path = "C:/Users/chan/poc-story-field-ai/travel_test.wav"
#     text = transcribe_audio(audio_path, model_size="medium")
#     print(text)
