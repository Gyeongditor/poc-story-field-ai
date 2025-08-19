import openai
import os
import requests

def make_image(story_result, keyword=None, save_dir='./result/gen_img') -> list:
    """
    story_result(페이지별 동화 내용 리스트)와 키워드를 받아, 각 페이지별로 일관된 화풍과 텍스트 없는 이미지를 생성
    """
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.')
    os.makedirs(save_dir, exist_ok=True)
    client = openai.OpenAI(api_key=api_key)
    image_paths = []
    keyword = keyword or {}
    # 스타일 관련 키워드 추출 및 기본값 지정
    atmosphere = keyword.get('atmosphere', '따뜻한 분위기')
    drawing_style = keyword.get('drawingStyle', '수채화')
    # 일관된 화풍 + 텍스트 금지 프롬프트 (언어/인종 관련 금지어 제거)
    style_suffix = (
        f", {drawing_style}, {atmosphere}, 동화책 일러스트, 일관된 화풍, soft lighting, pastel colors, "
        "storybook illustration, consistent style, same character design, watercolor, children's book, trending on artstation, "
        # 텍스트/워터마크/자막/서명 등만 금지
        "No text, no letters, no captions, no watermark, no signature, no writing, no subtitles, no handwriting, no calligraphy, no logo, no label, no typing, no printed text, "
        "글자, 텍스트, 자막, 워터마크, 서명, 문구, 문자는 절대 넣지 마"
    )
    for page in story_result:
        # 각 페이지별 프롬프트에 스타일 고정 문구 추가
        # instruction: 그림 생성을 위한 지시문
        instruction = "다음 내용을 바탕으로 그림을 그려줘. 핵심 인물과 행동에 집중해서 하나의 장면으로 그려줘."
        # 지시문 + 동화 내용 + 스타일/부정 키워드 조합
        prompt = f"{instruction} {page['content']}{style_suffix}"
        response = client.images.generate(
            model='dall-e-3',
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        img_data = requests.get(image_url).content
        save_path = os.path.join(save_dir, page["filename"])
        with open(save_path, 'wb') as handler:
            handler.write(img_data)
        image_paths.append(save_path)
    return image_paths


def make_thumbnail(title, keyword=None, save_dir='./result'):
    """
    동화 제목과 키워드를 받아 썸네일(표지) 이미지를 생성하고 경로 반환
    """
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.')
    os.makedirs(save_dir, exist_ok=True)
    client = openai.OpenAI(api_key=api_key)
    keyword = keyword or {}
    atmosphere = keyword.get('atmosphere', '따뜻한 분위기')
    drawing_style = keyword.get('drawingStyle', '수채화')
    # 표지에도 동일한 스타일/텍스트 금지 프롬프트 적용
    prompt = (
        f"{title}을 연상시키는 그림, {drawing_style}, {atmosphere}, 동화책 일러스트, "
        "일관된 화풍, soft lighting, pastel colors, children's book illustration, "
        "consistent style, same character design, trending on artstation, "
        "No text, no letters, no captions, no watermark, no signature, no writing, no subtitles, no handwriting, no calligraphy, no logo, no label, no typing, no printed text. Absolutely no words. "
        "글자, 텍스트, 자막, 워터마크, 서명, 문구, 문자는 절대 넣지 마"
    )
    response = client.images.generate(
        model='dall-e-3',
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response.data[0].url
    img_data = requests.get(image_url).content
    save_path = os.path.join(save_dir, 'title.png')
    with open(save_path, 'wb') as handler:
        handler.write(img_data)
    return save_path