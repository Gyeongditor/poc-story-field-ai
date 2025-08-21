FROM python:3.10-slim

# System deps for whisper (ffmpeg) and installing whisper from GitHub (git)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY fastapi/requirements.txt /app/requirements.txt

# Install Python deps (CUDA-enabled PyTorch) with BuildKit cache for pip
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio \
    && pip install -r requirements.txt

# Copy the rest of the application
COPY fastapi/ /app/

EXPOSE 8000

# Run main_whisper FastAPI app
CMD ["uvicorn", "main_whisper:app", "--host", "0.0.0.0", "--port", "8000"]
