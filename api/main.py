"""FastAPI service for pneumonia model inference."""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.config import load_config
from src.inference import predict_from_pil


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


CONFIG_PATH = _project_path(os.getenv("CONFIG_PATH", "config.yaml"))
CONFIG = load_config(CONFIG_PATH)


_default_model = str(
    Path(CONFIG["training"]["model_dir"]) / CONFIG["training"]["model_output_name"]
)
MODEL_PATH = _project_path(os.getenv("MODEL_PATH", _default_model))
IMG_SIZE = tuple(CONFIG["data"]["img_size"])
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", CONFIG["evaluation"]["threshold"]))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}

app = FastAPI(
    title="Pneumonia Detection API",
    version="1.0.0",
    description="Research demonstration only; not intended for clinical diagnosis.",
)


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    threshold: float


@app.get("/health")
def health() -> dict[str, str | bool]:
    available = MODEL_PATH.is_file()
    return {
        "status": "ready" if available else "model_missing",
        "model_available": available,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Supported formats: JPEG, PNG, BMP, and WebP.",
        )

    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {MAX_UPLOAD_BYTES}-byte upload limit.",
        )

    try:
        image = Image.open(BytesIO(payload))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image.") from exc

    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not available at {MODEL_PATH}.",
        )

    # TensorFlow inference is blocking; keep it off the asynchronous event loop.
    result = await run_in_threadpool(
        predict_from_pil,
        image=image,
        model_path=str(MODEL_PATH),
        img_size=IMG_SIZE,
        threshold=THRESHOLD,
    )
    return PredictionResponse(**result)
