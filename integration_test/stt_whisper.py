import whisper
import os

def transcribe(audio_path, model_size='medium', device=None):
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(audio_path)
    return result['text'] 