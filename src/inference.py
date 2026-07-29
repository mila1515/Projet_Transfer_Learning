"""Production inference services for operational usage.

This module executes image-level scoring with consistent preprocessing and
optional explainability outputs, enabling stable prediction behavior across
CLI, notebook, and API channels.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

@lru_cache(maxsize=2)
def load_model(model_path: str) -> tf.keras.Model:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")
    return tf.keras.models.load_model(model_file)


def preprocess_image(image: Image.Image, img_size: Tuple[int, int]) -> np.ndarray:
    image = image.convert("RGB").resize(img_size)
    image_array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(image_array, axis=0)


def predict_from_pil(
    image: Image.Image,
    model_path: str,
    img_size: Tuple[int, int] = (224, 224),
    threshold: float = 0.5,
) -> Dict[str, float | str]:
    model = load_model(model_path)
    x = preprocess_image(image, img_size)
    probability = float(model.predict(x, verbose=0)[0][0])
    label = "PNEUMONIA" if probability >= threshold else "NORMAL"

    return {
        "prediction": label,
        "probability": probability,
        "threshold": threshold,
    }


def predict_from_path(
    image_path: str,
    model_path: str,
    img_size: Tuple[int, int] = (224, 224),
    threshold: float = 0.5,
    gradcam_output: str | None = None,
    last_conv_layer_name: str | None = None,
) -> Dict[str, float | str]:
    image = Image.open(image_path)
    result = predict_from_pil(image, model_path, img_size, threshold)

    if gradcam_output:
        # Keep optional visualization dependencies out of the API startup path.
        from src.gradcam import make_gradcam_heatmap, save_gradcam_overlay

        model = load_model(model_path)
        x = preprocess_image(image, img_size)
        heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name)
        overlay_path = save_gradcam_overlay(image_path, heatmap, gradcam_output)
        result["gradcam_path"] = overlay_path

    return result
