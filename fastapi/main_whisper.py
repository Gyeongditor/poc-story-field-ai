from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import re
import io
import json
import os
from datetime import datetime
import zipfile
from api.gpt_casual2formal import casual2formal  # 구어체 → 문어체 변환
from api.gpt_story import make_story, make_title  # 동화 생성
from api.dalle_image import make_image, make_thumbnail  # 이미지 생성
from api.whisper import transcribe_audio  # STT 변환

# .env 파일 자동 로드 (OPENAI_API_KEY 등 환경변수 사용 가능)
load_dotenv()

app = FastAPI()

@app.post("/process")
async def process(
    data: str = Form(...),  # JSON string (character, age, sex, keyword)
    audio: UploadFile = File(...)
):
    """
    오디오 파일(wav, mp3 등) 업로드를 받아 STT 변환 후 동화/이미지 생성
    - data: JSON string (character, age, sex, keyword)
    - audio: wav/mp3 파일 업로드
    - 결과: story.txt, title.txt, 이미지 등 zip 파일로 반환
    """
    # 1. JSON 파싱
    data_dict = json.loads(data)
    character = data_dict.get("character")
    age = data_dict.get("age")
    sex = data_dict.get("sex")
    keyword_dict = data_dict.get("keyword", {})

    # 2. 오디오 파일 임시 저장
    audio_path = f"temp_{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    # 3. whisper로 STT 변환
    stt_text = transcribe_audio(audio_path, model_size="medium")

    # 4. 임시 파일 삭제
    os.remove(audio_path)

    # 5. storyContent에 STT 결과 사용
    data = {
        "character": character,
        "age": age,
        "sex": sex,
        "storyContent": stt_text,
        "keyword": keyword_dict
    }

    # 6. storyContent를 문어체로 변환
    formal_story = casual2formal(data["storyContent"])

    # 7. 페이지 구분(없으면 한 페이지로)
    def split_story_by_page(story: str) -> list:
        pattern = r"(?:\n|\r|\r\n)?\s*\d+페이지\s*:"
        splits = re.split(pattern, story)
        pages = [s.strip() for s in splits[1:]]
        return pages
    pages = split_story_by_page(formal_story)

    # 8. 동화 생성 정보 구성
    story_info = {
        "character": data.get("character"),
        "age": data.get("age"),
        "sex": data.get("sex"),
        "pages": pages,
        "keyword": data.get("keyword", {})
    }
    # 9. 동화 생성 (페이지별 내용)
    story_result = make_story(story_info)
    # 10. 동화 제목 생성
    title = make_title(story_result)

    # 11. 결과 저장 폴더 생성 (타임스탬프 기반)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("result", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    gen_img_dir = os.path.join(output_dir, "gen_img")
    os.makedirs(gen_img_dir, exist_ok=True)

    # 12. title.txt 저장
    title_txt_path = os.path.join(output_dir, "title.txt")
    with open(title_txt_path, "w", encoding="utf-8") as f:
        f.write(title)

    # 13. story.txt 저장 (N페이지:내용 형식, 안내문/마크다운/빈 줄 건너뜀, 본문만 연속 저장)
    story_txt_path = os.path.join(output_dir, "story.txt")
    valid_pages = []
    for page in story_result:
        content = page['content'].strip()
        if re.match(r'^\*+페이지\s*\d+\*+', content) or len(content) < 8:
            continue
        valid_pages.append(content)
    with open(story_txt_path, "w", encoding="utf-8") as f:
        page_idx = 1
        for content in valid_pages:
            clean_content = re.sub(r'[\[\]",\n]', '', content).strip()
            f.write(f"{page_idx}페이지:{clean_content}\n")
            page_idx += 1

    # 14. 이미지 생성 (프롬프트에 스타일/텍스트 금지 등 포함)
    thumbnail_path = make_thumbnail(title, story_info["keyword"], save_dir=output_dir)
    image_result = make_image(story_result, story_info["keyword"], save_dir=gen_img_dir)

    # 15. story.json 생성 (리스트만 포함)
    text_content = story_result
    story_json_path = os.path.join(output_dir, "story.json")
    with open(story_json_path, "w", encoding="utf-8") as f:
        json.dump(text_content, f, ensure_ascii=False, indent=2)

    # 16. 결과 zip 파일로 묶기
    zip_path = os.path.join(output_dir, "result.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file != "result.zip":
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname=arcname)

    # 17. zip 파일 반환
    return FileResponse(zip_path, filename="result.zip", media_type="application/zip")
