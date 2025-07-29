import openai
import os
import requests

def generate_images(prompts, api_key, save_dir='./integration_test/result/gen_img', model='dall-e-3'):
    os.makedirs(save_dir, exist_ok=True)
    client = openai.OpenAI(api_key=api_key)
    image_paths = []
    for idx, prompt in enumerate(prompts, 1):
        response = client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        img_data = requests.get(image_url).content
        save_path = os.path.join(save_dir, f'page_{idx}.png')
        with open(save_path, 'wb') as handler:
            handler.write(img_data)
        image_paths.append(save_path)
        print(f'[{idx}] 이미지 저장 완료:', save_path)
    return image_paths

# 예시 사용법 (main.py에서 호출)
# prompts = ["프롬프트1", "프롬프트2", ...]
# api_key = ...
# generate_images(prompts, api_key) 