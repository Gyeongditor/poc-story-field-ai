import whisper
import os

def transcribe(audio_path, model_size='medium', device='cpu'):
    os.environ["PATH"] += os.pathsep + r"C:\Users\chan\poc-story-field-ai\STT\whisper\ffmpeg\bin"
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(audio_path)
    return result['text'] 