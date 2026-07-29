FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONFIG_PATH=config_fast.yaml \
    MODEL_PATH=models/pneumonia_mobilenetv2_fast.keras

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt \
    && useradd --create-home --uid 10001 appuser

COPY config*.yaml ./
COPY src ./src
COPY api ./api
COPY models ./models

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import json, urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)); raise SystemExit(0 if data['model_available'] else 1)"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
