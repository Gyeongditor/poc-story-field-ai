"""
train.py
- Diffusers DreamBooth LoRA 학습
- accelerate launch 로 실행
"""

import argparse
from accelerate.utils import write_basic_config
from subprocess import run

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--train_text_encoder", action="store_true", help="스타일 LoRA는 보통 비활성")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    write_basic_config()  # accelerate default config

    MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    VAE   = "madebyollin/sdxl-vae-fp16-fix"

    cmd = [
        "accelerate", "launch",
        "train_dreambooth_lora_sdxl.py",  # diffusers/examples/dreambooth 에 있는 공식 스크립트
        f"--pretrained_model_name_or_path={MODEL}",
        f"--pretrained_vae_model_name_or_path={VAE}",
        f"--instance_data_dir={args.data_dir}",
        f"--output_dir={args.output_dir}",
        "--instance_prompt= ",  # 캡션(txt) 기반. 빈 프롬프트로 캡션 우선 사용
        f"--resolution={args.resolution}",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--learning_rate=1e-4",
        "--max_train_steps={}".format(args.steps),
        *( ["--train_text_encoder"] if args.train_text_encoder else [] ),
        "--snr_gamma=5.0",
        "--mixed_precision=fp16",
        "--enable_xformers_memory_efficient_attention",
        "--gradient_checkpointing",
        "--use_8bit_adam"
    ]
    # 재현성 및 체크포인트 저장 권장 옵션
    cmd.extend([
        f"--seed={args.seed}",
        "--checkpointing_steps=1000",
        "--save_steps=1000",
    ])
    run(cmd)

if __name__ == "__main__":
    main()
