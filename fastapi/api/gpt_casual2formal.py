import openai
import os

def casual2formal(text: str) -> str:
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError('OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.')
    client = openai.OpenAI(api_key=api_key)
    prompt = f"다음 구어체 문장을 문어체로 바꿔줘:\n{text}"
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content
