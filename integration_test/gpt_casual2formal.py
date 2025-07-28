import openai
import os

def casual_to_formal(text, api_key):
    client = openai.OpenAI(api_key=api_key)
    prompt = f"다음 구어체 문장을 문어체로 바꿔줘:\n{text}"
    response = client.chat.completions.create(
         #model='gpt-3.5-turbo'
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content 