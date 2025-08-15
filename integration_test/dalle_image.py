import openai
import os
import requests

def generate_images(prompts, api_key, save_dir='./result/gen_img', model='dall-e-3'):
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

def generate_title_image(prompt, api_key, save_dir='./result', model='dall-e-3'):
    import requests
    import os
    os.makedirs(save_dir, exist_ok=True)
    client = openai.OpenAI(api_key=api_key)
    response = client.images.generate(
        model=model,
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response.data[0].url
    img_data = requests.get(image_url).content
    save_path = os.path.join(save_dir, 'title.png')
    with open(save_path, 'wb') as handler:
        handler.write(img_data)
    print(f'title.png 표지 이미지 저장 완료: {save_path}')
    return save_path
