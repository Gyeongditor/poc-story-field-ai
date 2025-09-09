from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# 모델 로드
model_path = "./outputs"
base_model = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 테스트 프롬프트
prompt = """캐릭터: 토끼
나이: 5살
성별: 여
분위기: 따뜻한
그림체: 수채화
내용: 토끼가 숲에서 친구를 만났습니다.

다음 조건을 반영하여 5~10페이지, 페이지당 2~3문장의 동화를 작성해줘.
"""

result = pipe(prompt, max_new_tokens=700, temperature=0.7, top_p=0.9)
print(result[0]["generated_text"])
