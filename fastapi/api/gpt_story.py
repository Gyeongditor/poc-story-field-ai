import openai
import os
import re

def make_story(story_info: dict) -> list:
    """
    story_info(dict)로부터 동화 페이지별 내용을 생성
    - storyContent의 실제 내용을 동화에 반드시 반영하도록 프롬프트에 명시
    - 5~8페이지, 'N페이지:내용' 한 줄 형식으로만 응답하도록 강하게 요구
    - 각 페이지는 3~6문장, 200~400자 분량, 풍부한 묘사/대화/감정/동화적 분위기/상상력/배경/생각/느낌 포함
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
            "아래 정보를 바탕으로 동화책을 5~8페이지로 나눠서 각 페이지별로 3~6문장, 200~400자 분량으로 써줘. "
            "각 페이지는 구체적인 상황 묘사, 대화, 감정, 동화적 분위기, 상상력, 배경 설명, 캐릭터의 생각과 느낌까지 풍부하게 써. "
            "짧고 단순하게 쓰지 말고, 진짜 동화책 한 페이지처럼 써줘. "
            "!!! 반드시 아래 형식으로만 응답해줘. 안내문, 코드, 리스트, 마크다운, 설명, 인삿말, 기타 부가 텍스트는 절대 넣지 마. "
            "형식 예시: \n"
            "1페이지:옛날 옛적, 작은 마을에 늘 여행을 꿈꾸던 소녀 하나가 살고 있었어요. 그녀는 하늘 높이 떠다니는 구름을 보며 그곳에 가보고 싶었고, 푸른 바다를 바라보며 물속의 신비한 생물들을 만져보고 싶다는 소망을 품었답니다. 그러나 마을 사람들이 언제나 그녀에게 “여행은 위험해!”라고 말하며 꿈을 꺾곤 했어요.\n"
            "2페이지: ...\n"
            "이런 식으로, 각 페이지는 'N페이지:내용' 한 줄로만, 총 5~8줄로만 응답해. "
            f"- 등장인물: {character} ({age}세, {sex})\n"
            f"- 키워드: {keyword}\n"
            f"- storyContent: {pages[0]}\n"
            f"{must_use_content_kr}\n{must_use_content_en}\n"
            "Each page should be 3~6 sentences, 200~400 characters, with rich description, dialogue, emotions, fairy-tale atmosphere, imagination, background, and the character's thoughts and feelings. Do not write short or simple sentences."
        )
    else:
        prompt = (
            "아래 정보를 바탕으로 동화책의 각 페이지 내용을 만들어줘.\n"
            "각 페이지는 3~6문장, 200~400자 분량, 구체적인 상황 묘사, 대화, 감정, 동화적 분위기, 상상력, 배경 설명, 캐릭터의 생각과 느낌까지 풍부하게 써. "
            "짧고 단순하게 쓰지 말고, 진짜 동화책 한 페이지처럼 써줘. "
            "!!! 반드시 아래 형식으로만 응답해줘. 안내문, 코드, 리스트, 마크다운, 설명, 인삿말, 기타 부가 텍스트는 절대 넣지 마. "
            "형식 예시: \n"
            "1페이지:옛날 옛적, 작은 마을에 늘 여행을 꿈꾸던 소녀 하나가 살고 있었어요. 그녀는 하늘 높이 떠다니는 구름을 보며 그곳에 가보고 싶었고, 푸른 바다를 바라보며 물속의 신비한 생물들을 만져보고 싶다는 소망을 품었답니다. 그러나 마을 사람들이 언제나 그녀에게 “여행은 위험해!”라고 말하며 꿈을 꺾곤 했어요.\n"
            "2페이지: ...\n"
            "이런 식으로, 각 페이지는 'N페이지:내용' 한 줄로만, 총 5~8줄로만 응답해. "
            f"- 등장인물: {character} ({age}세, {sex})\n"
            f"- 키워드: {keyword}\n"
            "- 각 페이지별 내용(구체적 상황):\n"
        )
        for idx, page in enumerate(pages, 1):
            prompt += f"{idx}페이지:{page}\n"
        prompt += (
            f"{must_use_content_kr}\n{must_use_content_en}\n"
            "Each page should be 3~6 sentences, 200~400 characters, with rich description, dialogue, emotions, fairy-tale atmosphere, imagination, background, and the character's thoughts and feelings. Do not write short or simple sentences."
        )
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
