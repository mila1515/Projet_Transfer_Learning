"""Shared operational utilities for analytics and ML workflows.

This module centralizes reproducibility, artifact management, and reporting
helpers to keep experiment outputs consistent and decision-ready.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: str | Path) -> Path:
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_json(payload: Dict, output_path: str | Path) -> None:
    path = Path(output_path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=CLASS_NAMES,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def collect_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    file_paths: Iterable[str],
    max_items: int = 20,
) -> Dict[str, List[str]]:
    fp, fn = [], []
    for label, pred, image_path in zip(y_true.astype(int), y_pred.astype(int), file_paths):
        if pred == 1 and label == 0 and len(fp) < max_items:
            fp.append(image_path)
        if pred == 0 and label == 1 and len(fn) < max_items:
            fn.append(image_path)
    return {"false_positives": fp, "false_negatives": fn}
