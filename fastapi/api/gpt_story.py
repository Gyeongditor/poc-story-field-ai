import openai
import os
import re

def make_story(story_info: dict) -> list:
    """
    story_info(dict)로부터 동화 페이지별 내용을 생성
    - storyContent의 실제 내용을 동화에 반드시 반영하도록 프롬프트에 명시
    - 5~8페이지, 'N페이지:내용' 한 줄 형식으로만 응답하도록 강하게 요구
    """
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.')
    client = openai.OpenAI(api_key=api_key)
    character = story_info.get('character', '')
    age = story_info.get('age', '')
    sex = story_info.get('sex', '')
    pages = story_info.get('pages', [])
    keyword = story_info.get('keyword', {})
    # storyContent를 반드시 동화에 반영하라고 명시 (한글+영어)
    must_use_content_kr = (
        "**아래 storyContent의 실제 내용을 동화에 반드시 반영해. 등장인물, 사건, 장소, 대화, 감정 등 구체적으로 활용해. 무시하지 마!**"
    )
    must_use_content_en = (
        "You must use the actual storyContent in the story. Do not ignore it. Use the character, events, places, conversations, emotions, and details from storyContent."
    )
    # 프롬프트 조합
    if len(pages) == 1:
        prompt = (
            "아래 정보를 바탕으로 동화책을 5~8페이지로 나눠서 각 페이지별로 2~3문장씩 동화체로 써줘. "
            "!!! 반드시 아래 형식으로만 응답해줘. 안내문, 코드, 리스트, 마크다운, 설명, 인삿말, 기타 부가 텍스트는 절대 넣지 마. "
            "형식 예시: \n"
            "1페이지:병아리는 숲을 산책했어요.\n"
            "2페이지:병아리는 친구를 만났어요.\n"
            "3페이지:병아리는 집에 돌아왔어요.\n"
            "이런 식으로, 각 페이지는 'N페이지:내용' 한 줄로만, 총 5~8줄로만 응답해. "
            f"- 등장인물: {character} ({age}세, {sex})\n"
            f"- 키워드: {keyword}\n"
            f"- storyContent: {pages[0]}\n"
            f"{must_use_content_kr}\n{must_use_content_en}"
        )
    else:
        prompt = (
            "아래 정보를 바탕으로 동화책의 각 페이지 내용을 만들어줘.\n"
            "!!! 반드시 아래 형식으로만 응답해줘. 안내문, 코드, 리스트, 마크다운, 설명, 인삿말, 기타 부가 텍스트는 절대 넣지 마. "
            "형식 예시: \n"
            "1페이지:병아리는 숲을 산책했어요.\n"
            "2페이지:병아리는 친구를 만났어요.\n"
            "3페이지:병아리는 집에 돌아왔어요.\n"
            "이런 식으로, 각 페이지는 'N페이지:내용' 한 줄로만, 총 5~8줄로만 응답해. "
            f"- 등장인물: {character} ({age}세, {sex})\n"
            f"- 키워드: {keyword}\n"
            "- 각 페이지별 내용(구체적 상황):\n"
        )
        for idx, page in enumerate(pages, 1):
            prompt += f"{idx}페이지:{page}\n"
        prompt += f"{must_use_content_kr}\n{must_use_content_en}"
    # OpenAI API 호출
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}]
    )
    result = response.choices[0].message.content
    # 페이지별로 분리: 'N페이지:'로 분리
    page_splits = re.split(r'\n?\s*\d+페이지\s*[:：]', result)
    # 첫 번째는 빈 문자열이므로 제외
    page_splits = [p.strip() for p in page_splits if p.strip()]
    # 5~8페이지로 제한
    page_splits = page_splits[:8]
    if len(page_splits) < 5:
        page_splits += ["(빈 페이지)"] * (5 - len(page_splits))
    # [{pageNumber, content, filename}, ...] 형식으로 반환
    return [
        {
            "pageNumber": idx + 1,
            "content": text,
            "filename": f"page_{idx+1}.png"
        }
        for idx, text in enumerate(page_splits)
    ]

def make_title(story_result: list) -> str:
    """
    동화 전체 내용을 받아 제목을 생성
    """
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.')
    client = openai.OpenAI(api_key=api_key)
    # 각 페이지의 content만 추출해서 하나의 문자열로 합침
    story_text = '\n'.join([page['content'] for page in story_result])
    prompt = f"다음 동화의 내용을 보고, 어울리는 한국어 제목을 한 문장으로 지어줘.\n\n{story_text}"
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content.strip()
