"""Prediction entry points for analyst and engineering workflows.

This module offers reusable helpers and a CLI command to run single-image
inference, supporting quick validation, demos, and operational checks.
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

from PIL import Image

from src.inference import predict_from_path, predict_from_pil


def predict_pil_image(
    image: Image.Image,
    model_path: str,
    img_size: Tuple[int, int] = (224, 224),
    threshold: float = 0.5,
) -> Dict[str, float | str]:
    return predict_from_pil(
        image=image,
        model_path=model_path,
        img_size=img_size,
        threshold=threshold,
    )


def predict_image(
    image_path: str,
    model_path: str,
    img_size: Tuple[int, int] = (224, 224),
    threshold: float = 0.5,
    save_gradcam: bool = False,
    gradcam_output_path: str | None = None,
    last_conv_layer_name: str | None = None,
) -> Dict[str, float | str]:
    return predict_from_path(
        image_path=image_path,
        model_path=model_path,
        img_size=img_size,
        threshold=threshold,
        gradcam_output=gradcam_output_path if save_gradcam else None,
        last_conv_layer_name=last_conv_layer_name,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict pneumonia from a single chest X-ray image.")
    parser.add_argument("--image", type=str, required=True, help="Path to X-ray image.")
    parser.add_argument("--model", type=str, default="models/pneumonia_model.keras", help="Path to model file.")
    parser.add_argument("--img-size", type=int, nargs=2, default=(224, 224), help="Input image size.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold.")
    parser.add_argument("--gradcam", action="store_true", help="Save Grad-CAM overlay.")
    parser.add_argument("--gradcam-output", type=str, default="reports/figures/gradcam_overlay.png")
    parser.add_argument("--last-conv-layer", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = predict_image(
        image_path=args.image,
        model_path=args.model,
        img_size=tuple(args.img_size),
        threshold=args.threshold,
        save_gradcam=args.gradcam,
        gradcam_output_path=args.gradcam_output,
        last_conv_layer_name=args.last_conv_layer,
    )
    print(result)


if __name__ == "__main__":
    main()
