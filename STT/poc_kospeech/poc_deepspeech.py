# import torch
# from kospeech.models.deepspeech2 import DeepSpeech2
# from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
# from kospeech.data.audio.processor import SpectrogramParser
# from kospeech.opt import load_config

# # 경로 설정
# model_path = 'deepspeech2.pt'  # 사전 학습 모델 다운로드 필요
# config_path = 'config.yaml'
# audio_path = '/mnt/data/90457_여행.wav'

# # 디바이스 설정
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # config 및 vocab 로드
# config = load_config(config_path)
# vocab = KsponSpeechVocabulary()
# parser = SpectrogramParser(audio_config=config['audio'], vocab=vocab, feature_extract_by='torchaudio')

# # 오디오 특징 추출
# features = parser.parse(audio_path).unsqueeze(0).to(device)

# # 모델 로드
# model = DeepSpeech2(vocab, config['model'])
# checkpoint = torch.load(model_path, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval().to(device)

# # 예측
# with torch.no_grad():
#     output = model(features)
#     decoded = vocab.label_to_string(output.argmax(dim=-1))

# print("🗣️ 인식된 텍스트:", decoded[0])
import torch
print(torch.__version__)