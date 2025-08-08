from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from diffusers import StableDiffusionPipeline
import torch

def generate_story():
    model_id = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    prompt = "Write a short bedtime story for children about a rabbit and a magical forest:"
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    story = generator(prompt, max_new_tokens=300, do_sample=True)[0]["generated_text"]
    return story

def extract_sentence(story: str):
    lines = [line.strip() for line in story.split('.') if len(line.strip()) > 20]
    return lines[0] + "." if lines else "A magical forest with animals."

def generate_image(prompt: str):
    print("🎨 Generating image for:", prompt)
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-1-5",
        torch_dtype=torch.float16
    ).to("cuda")
    image = pipe(prompt).images[0]
    image.save("storybook_image.png")
    print("✅ Image saved as storybook_image.png")

if __name__ == "__main__":
    print("📖 Generating story...")
    story = generate_story()
    print("📜 Story:\n", story)

    sentence = extract_sentence(story)
    print("🖼️ Prompt:", sentence)

    generate_image(sentence)
