FROM continuumio/miniconda3

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN conda create --name myenv python=3.10 -y && \
    conda clean -afy && \
    /bin/bash -c "source activate myenv && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121"


SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

WORKDIR /app
COPY . .

CMD ["conda", "run", "-n", "myenv", "python", "test.py"]
