import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from gpt_casual2formal import casual_to_formal
from gpt_story import generate_story, save_story
from dalle_image import generate_images, generate_title_image

app = FastAPI()

class StoryRequest(BaseModel):
    stt: str
    keyword: Optional[List[str]] = None

@app.post("/process")
async def process_story(req: StoryRequest):
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    # 1. 구어체 → 문어체 변환
    casual_text = casual_to_formal(req.stt, api_key)

    # 2. 동화 줄거리 생성 프롬프트
    story_prompt = (
        "다음 내용을 바탕으로 5페이지 분량의 동화를 써줘. "
        "각 페이지의 시작은 반드시 [PAGE]로 시작하게 하고, [PAGE]가 5번 등장해야 해. "
        "각 페이지는 3~5문장으로 구성해줘. "
        "예시:\n[PAGE] ...\n[PAGE] ...\n[PAGE] ...\n[PAGE] ...\n[PAGE] ...\n"
        "주인공이 명확한 아이들을 위한 동화였으면 좋겠어.\n"
        f"{casual_text}"
    )
    if req.keyword:
        story_prompt += "\n" + ", ".join(req.keyword)

    # 3. 동화 줄거리 생성
    story = generate_story(story_prompt, api_key)
    # 3-1. 동화 제목 생성
    title_prompt = (
        "다음 동화의 제목을 한 줄로 지어줘.\n"
        f"{story}"
    )
    title = generate_story(title_prompt, api_key).strip().replace('\n', '')
    # 제목 저장
    title_save_path = './result/title.txt'
    os.makedirs(os.path.dirname(title_save_path), exist_ok=True)
    with open(title_save_path, 'w', encoding='utf-8') as f:
        f.write(title)
    # 4. 동화 줄거리 파일로 저장
    save_story(story, "story.txt")
    # 5. [PAGE] 기준 분리
    story_pages = [s.strip() for s in story.split('[PAGE]') if s.strip()]

    # 6. 각 페이지별 이미지 프롬프트 생성
    image_prompts = [
        f"동화의 한 장면을 삽화 스타일로 그려줘. 그림에는 어떠한 글자나 텍스트가 들어가지 않게 해줘. "
        f"밝고 따뜻한 색감의 일러스트, 통일된 동화책 스타일, 그림들이 통일된 화풍을 가졌으면 좋겠어., {page[:100]}"
        for page in story_pages
    ]
    # 7. 이미지 생성
    image_paths = generate_images(image_prompts, api_key)

    # 8. 표지 이미지 프롬프트 및 생성
    cover_prompt = (
        f"이 동화의 표지 이미지를 그려줘. 밝고 따뜻한 색감, 통일된 동화책 스타일, 그림에는 글자나 텍스트가 들어가지 않게 해줘. 제목: {title}"
    )
    cover_image_path = generate_title_image(cover_prompt, api_key, save_dir='./result', model='dall-e-3')

    return {
        "title": title,
        "cover_image_path": cover_image_path,
        "pages": story_pages,
        "image_paths": image_paths
    }
