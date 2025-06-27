import whisper
import torch
import time
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
start_time = time.time()
small_model = whisper.load_model('small',device=device)
#audio_path = f"./fairytale_test.mp3"
audio_path = f"./travel_test.wav"
result = small_model.transcribe(audio_path,language="ko",fp16=False)
print(result['text'])

end_time = time.time()
elapsed_time = end_time - start_time
print(f"총 소요 시간: {elapsed_time:.2f}초")