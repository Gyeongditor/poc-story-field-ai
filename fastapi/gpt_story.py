import openai
import os

def generate_story(prompt, api_key):
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        #model='gpt-3.5-turbo'
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response.choices[0].message.content

def save_story(story, filename, save_dir="./result/gen_story"):
    import os
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(story)
    print(f"스토리가 저장되었습니다: {file_path}")
