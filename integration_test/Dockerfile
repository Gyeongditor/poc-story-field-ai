# FROM python:3.10-slim

# # ffmpeg 설치
# RUN apt-get update && \
#     apt-get install -y ffmpeg && \
#     rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # 코드 복사
# COPY ../integration_test/ ./integration_test/
# COPY requirements.txt .

# # 패키지 설치
# RUN pip install --upgrade pip && pip install -r requirements.txt

# # 환경변수 예시 (실제 키는 --env-file 또는 .env로 관리 권장)
# # ENV OPENAI_API_KEY=your_openai_api_key

# CMD ["python", "integration_test/main.py"] 

FROM python:3.10-slim

# FFmpeg 설치
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉터리 설정
WORKDIR /app

# 현재 디렉터리 전체 복사
COPY . .

# Python 패키지 설치
RUN pip install --upgrade pip && pip install -r requirements.txt

# 실행 커맨드 (main.py 기준)
CMD ["python", "main.py"]

