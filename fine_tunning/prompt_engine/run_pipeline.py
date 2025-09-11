import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stt_file", default="STT.txt", help="STT 원문 텍스트 파일 경로")
    parser.add_argument("--keywords", required=True, help="동화 분위기 키워드")
    parser.add_argument("--character", required=True, help="주인공 정보")
    parser.add_argument("--summary_prompt_file", default="summarize_prompt.txt")
    parser.add_argument("--story_prompt_file", default="story_prompt.txt")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--summary_tokens", type=int, default=400)
    parser.add_argument("--story_tokens", type=int, default=2000)
    parser.add_argument("--summary_output", default="summary_output.txt")
    parser.add_argument("--story_output", default="story_output.txt")
    args = parser.parse_args()

    # 1) 요약 실행
    summarize_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "summarize.py"),
        "--input_file", args.stt_file,
        "--prompt_file", args.summary_prompt_file,
        "--output_file", args.summary_output,
        "--model_id", args.model_id,
        "--max_tokens", str(args.summary_tokens),
    ]
    print("\n[1/2] STT 요약 실행 중...\n")
    subprocess.run(" ".join(summarize_cmd), shell=True, check=True)

    # 2) 동화 생성 실행
    prompt_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "prompt.py"),
        "--keywords", args.keywords,
        "--character", args.character,
        "--summary_file", args.summary_output,
        "--prompt_file", args.story_prompt_file,
        "--model_id", args.model_id,
        "--max_tokens", str(args.story_tokens),
        "--output_file", args.story_output,
    ]
    print("\n[2/2] 요약 기반 동화 생성 중...\n")
    subprocess.run(" ".join(prompt_cmd), shell=True, check=True)

    print(f"\n완료: 동화 결과는 {args.story_output} 에 저장되었습니다.")


if __name__ == "__main__":
    main()


