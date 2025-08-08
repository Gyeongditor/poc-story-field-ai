import os
from dotenv import load_dotenv
from stt_whisper import transcribe
from gpt_casual2formal import casual_to_formal
from gpt_story import generate_story, save_story
from dalle_image import generate_images
import torch

def main():
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    audio_path = 'C:/Users/chan/poc-story-field-ai/integration_test/travel_test.wav'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stt_text = transcribe(audio_path, device=device)  # GPU 사용시 'cuda'
    print('[STT 결과]', stt_text)

    casual_text = casual_to_formal(stt_text, api_key)
    print('[문어체 변환 결과]', casual_text)

    story_prompt = (
        "다음 내용을 바탕으로 5페이지 분량의 동화를 써줘. "
        "각 페이지의 시작은 반드시 [PAGE]로 시작하게 하고, [PAGE]가 5번 등장해야 해. "
        "각 페이지는 3~5문장으로 구성해줘. "
        "예시:\n[PAGE] ...\n[PAGE] ...\n[PAGE] ...\n[PAGE] ...\n[PAGE] ...\n"
        "주인공이 명확한 아이들을 위한 동화였으면 좋겠어.\n"
        f"{casual_text}"
    )
    story = generate_story(story_prompt, api_key)
    print('[동화 생성 결과]', story)
    print(f"[PAGE] 등장 횟수: {story.count('[PAGE]')}")
    save_story(story, "story.txt")

    # [PAGE] 기준으로 분리
    story_pages = [s.strip() for s in story.split('[PAGE]') if s.strip()]
    print(f"[페이지 수]: {len(story_pages)}")

    image_prompts = [
        f"동화의 한 장면을 삽화 스타일로 그려줘. 그림에는 글자나 텍스트가 들어가지 않게 해줘. "
        f"밝고 따뜻한 색감의 일러스트, 통일된 동화책 스타일, 그림들이 통일된 화풍을 가졌으면 좋겠어., {page[:100]}"
        for page in story_pages
    ]
    image_paths = generate_images(image_prompts, api_key)
    print('[이미지 생성 완료]', image_paths)

if __name__ == '__main__':
    main() 