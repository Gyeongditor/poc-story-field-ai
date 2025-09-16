"""
train.py
- Diffusers DreamBooth LoRA 학습
- accelerate launch 로 실행
"""

import argparse
from diffusers import AutoencoderKL
from accelerate.utils import write_basic_config
from subprocess import run

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--steps", type=int, default=800)
    args = p.parse_args()

    write_basic_config()  # accelerate default config

    MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    VAE   = "madebyollin/sdxl-vae-fp16-fix"
    prompt = f"storybook illustration of {args.token}, watercolor, pastel colors, thin ink lines"

    cmd = [
        "accelerate", "launch",
        "train_dreambooth_lora_sdxl.py",  # diffusers/examples/dreambooth 에 있는 공식 스크립트
        f"--pretrained_model_name_or_path={MODEL}",
        f"--pretrained_vae_model_name_or_path={VAE}",
        f"--instance_data_dir={args.data_dir}",
        f"--output_dir={args.output_dir}",
        f"--instance_prompt={prompt}",
        "--resolution=1024",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--learning_rate=1e-4",
        "--max_train_steps={}".format(args.steps),
        "--train_text_encoder",
        "--snr_gamma=5.0",
        "--mixed_precision=fp16",
        "--enable_xformers_memory_efficient_attention",
        "--gradient_checkpointing",
        "--use_8bit_adam"
    ]
    run(cmd)

if __name__ == "__main__":
    main()
