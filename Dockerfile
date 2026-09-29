# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Forest Cover Type Classifier — inference API + web UI (CPU image)
#   docker build -t covtype-api .
#   docker run --rm -p 8000:8000 covtype-api
# ---------------------------------------------------------------------------
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    COVTYPE_DEVICE=cpu

WORKDIR /app

# Dependencies first for better layer caching (CPU-only PyTorch keeps the image small)
COPY requirements-api.txt .
RUN pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements-api.txt

# Application code and exported model artifacts
COPY covtype/ covtype/
COPY backend/ backend/
COPY frontend/ frontend/
COPY artifacts/ artifacts/

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
