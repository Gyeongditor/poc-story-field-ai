# syntax=docker/dockerfile:1.4

########################################
# 1단계: Python 패키지 빌드
########################################
FROM python:3.12.3-slim as builder

WORKDIR /app

# 필수 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 후 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


########################################
# 2단계: 런타임 환경
########################################
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

# Python 3.12 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# python 명령을 Python 3.12로 연결
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/hf-cache \
    TRANSFORMERS_CACHE=/hf-cache \
    TORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

# 빌드 단계에서 설치한 Python 패키지 복사
COPY --from=builder /install /usr/local

# 소스 코드 복사
COPY storybook_poc.py transcript.txt ./

# 기본 실행 명령
CMD ["python", "storybook_poc.py"]
