FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=6 \
    MKL_NUM_THREADS=6 \
    HF_HOME=/app/cache/huggingface \
    GRADIO_TEMP_DIR=/app/cache/gradio \
    TEMP=/app/cache/temp \
    TMP=/app/cache/temp \
    TMPDIR=/app/cache/temp \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-cloud.txt /app/requirements-cloud.txt
RUN pip install --no-cache-dir -r /app/requirements-cloud.txt

COPY . /app
RUN pip install --no-cache-dir --no-deps -e . \
    && mkdir -p /app/cache/huggingface /app/cache/gradio /app/cache/temp /app/outputs /app/voices /app/logs /app/temp /app/debug /app/benchmarks \
    && chmod -R 777 /app

EXPOSE 7860

CMD ["python", "app.py"]
