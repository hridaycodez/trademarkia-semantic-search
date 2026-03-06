# Single stage build for simplicity and compatibility
FROM python:3.11-slim-bullseye

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fix torch to CPU-only version
RUN pip uninstall torch -y && \
    pip install torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Install onnxruntime for chromadb compatibility
RUN pip install onnxruntime

COPY . .

RUN mkdir -p data models

ENV PYTHONUNBUFFERED=1
ENV VECTOR_STORE_DIR=/app/data/vecstore
ENV MODELS_DIR=/app/models
ENV CACHE_THRESHOLD=0.85

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
