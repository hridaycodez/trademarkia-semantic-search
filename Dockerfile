# ── Stage 1: builder ─────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source
COPY . .

# Create directories for persistent data
RUN mkdir -p data models

# Pre-download the embedding model so the container is self-contained
# (skipped if running offline; the model will download on first request)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

ENV PYTHONUNBUFFERED=1
ENV CHROMA_PERSIST_DIR=/app/data/chroma
ENV MODELS_DIR=/app/models
ENV CACHE_THRESHOLD=0.85

EXPOSE 8000

# The index must be built (via build_index.py) before starting.
# Mount /app/data and /app/models as volumes if you want persistence.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
