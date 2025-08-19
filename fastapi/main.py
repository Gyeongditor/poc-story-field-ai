from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict
from dotenv import load_dotenv  # .env 파일 자동 로드용
import re
import io
import json
import os
from datetime import datetime
import zipfile
from api.gpt_casual2formal import casual2formal
from api.gpt_story import make_story, make_title
from api.dalle_image import make_image, make_thumbnail

# .env 파일 자동 로드 (OPENAI_API_KEY 등 환경변수 사용 가능)
load_dotenv()

app = FastAPI()

# keyword 필드용 Pydantic 모델
class Keyword(BaseModel):
    atmosphere: Optional[str] = None  # 동화 분위기(예: 따뜻한)
    drawingStyle: Optional[str] = None  # 그림체(예: 수채화)

# 전체 입력값용 Pydantic 모델
class StoryRequest(BaseModel):
    character: str  # 등장인물 이름
    age: int        # 등장인물 나이
    sex: str        # 등장인물 성별
    storyContent: str  # 동화 원본 내용(긴 문장 또는 페이지 구분 없이)
    keyword: Optional[Keyword] = None  # (선택) 분위기, 그림체 등

def split_story_by_page(story: str) -> list:
    """입력 storyContent에서 'N페이지:' 구분이 있을 경우 페이지별로 분리"""
    pattern = r"(?:\n|\r|\r\n)?\s*\d+페이지\s*:"
    splits = re.split(pattern, story)
    page_numbers = re.findall(r"\d+페이지\s*:", story)
    pages = [s.strip() for s in splits[1:]]
    return pages

def make_story_list(story_result: list) -> list:
    """story_result를 [{pageNumber, content, filename}, ...] 리스트로 변환"""
    return [
        {
            "pageNumber": idx + 1,
            "content": text,
            "filename": f"page_{idx+1}.png"
        }
        for idx, text in enumerate(story_result)
    ]

@app.post("/process")
async def process(request: StoryRequest):
    """
    동화 생성 및 이미지 생성 API
    - 입력값: character, age, sex, storyContent, keyword
    - 결과: story.json, title.png, page_1.png, ... (zip 파일로 반환)
    """
    data = request.dict()
    # 1. storyContent를 문어체로 변환
    formal_story = casual2formal(data["storyContent"])
    # 2. 페이지 구분(없으면 한 페이지로)
    pages = split_story_by_page(formal_story)

    # 3. 동화 생성 정보 구성
    story_info = {
        "character": data.get("character"),
        "age": data.get("age"),
        "sex": data.get("sex"),
        "pages": pages,
        "keyword": data.get("keyword", {})
    }
    # 4. 동화 생성 (페이지별 내용)
    story_result = make_story(story_info)
    # 5. 동화 제목 생성
    title = make_title(story_result)

    # 6. 결과 저장 폴더 생성 (타임스탬프 기반)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("result", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    gen_img_dir = os.path.join(output_dir, "gen_img")
    os.makedirs(gen_img_dir, exist_ok=True)

    # 7. title.txt 저장
    title_txt_path = os.path.join(output_dir, "title.txt")
    with open(title_txt_path, "w", encoding="utf-8") as f:
        f.write(title)

    # 8. story.txt 저장 (N페이지:내용 형식, 안내문/마크다운/빈 줄 건너뜀, 본문만 연속 저장)
    story_txt_path = os.path.join(output_dir, "story.txt")
    valid_pages = []
    for page in story_result:
        content = page['content'].strip()
        # '**페이지 N**' 등 안내문/마크다운/빈 줄은 건너뜀
        if re.match(r'^\*+페이지\s*\d+\*+', content) or len(content) < 8:
            continue
        valid_pages.append(content)
    with open(story_txt_path, "w", encoding="utf-8") as f:
        page_idx = 1
        for content in valid_pages:
            clean_content = re.sub(r'[\[\]",\n]', '', content).strip()
            f.write(f"{page_idx}페이지:{clean_content}\n")
            page_idx += 1

    # 9. 이미지 생성 (프롬프트에 스타일/텍스트 금지 등 포함)
    thumbnail_path = make_thumbnail(title, story_info["keyword"], save_dir=output_dir)
    image_result = make_image(story_result, story_info["keyword"], save_dir=gen_img_dir)

    # 10. story.json 생성 (리스트만 포함)
    text_content = story_result
    story_json_path = os.path.join(output_dir, "story.json")
    with open(story_json_path, "w", encoding="utf-8") as f:
        json.dump(text_content, f, ensure_ascii=False, indent=2)

    # 11. 결과 zip 파일로 묶기
    zip_path = os.path.join(output_dir, "result.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file != "result.zip":
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname=arcname)

    # 12. zip 파일 반환
    return FileResponse(zip_path, filename="result.zip", media_type="application/zip")
