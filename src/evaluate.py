"""Evaluation and KPI reporting for model decision-making.

This module transforms predictions into business-facing indicators (accuracy,
recall, specificity, AUC) and reporting artifacts used for model validation,
risk analysis, and go/no-go decisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from src.config import load_config
from src.data_loader import build_datasets, list_split_image_paths
from src.utils import collect_errors, ensure_dir, plot_confusion_matrix, plot_roc_curve, save_json


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(auc(fpr, tpr)),
        "specificity": float(((y_true == 0) & (y_pred == 0)).sum() / max((y_true == 0).sum(), 1)),
    }

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["NORMAL", "PNEUMONIA"],
        output_dict=True,
        zero_division=0,
    )

    return {"metrics": metrics, "report": report_dict, "y_pred": y_pred}


def generate_evaluation_artifacts(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    report_dir: str | Path,
    file_paths: list[str] | None = None,
) -> Dict[str, str]:
    report_dir = ensure_dir(report_dir)
    figures_dir = ensure_dir(Path(report_dir) / "figures")

    result = evaluate_predictions(y_true, y_prob, threshold)
    y_pred = result["y_pred"]

    metrics_path = Path(report_dir) / "metrics.json"
    report_path = Path(report_dir) / "classification_report.json"
    errors_path = Path(report_dir) / "error_analysis.json"
    confusion_path = Path(figures_dir) / "confusion_matrix.png"
    roc_path = Path(figures_dir) / "roc_curve.png"

    save_json(result["metrics"], metrics_path)
    save_json(result["report"], report_path)

    if file_paths:
        error_cases = collect_errors(y_true, y_pred, file_paths, max_items=30)
        save_json(error_cases, errors_path)

    plot_confusion_matrix(y_true, y_pred, confusion_path)
    plot_roc_curve(y_true, y_prob, roc_path)

    return {
        "metrics": str(metrics_path),
        "classification_report": str(report_path),
        "error_analysis": str(errors_path),
        "confusion_matrix": str(confusion_path),
        "roc_curve": str(roc_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model on test split.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Optional model path override.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    data_cfg = config["data"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]

    model_path = args.model or str(Path(train_cfg["model_dir"]) / train_cfg["model_output_name"])
    model = tf.keras.models.load_model(model_path)

    datasets = build_datasets(
        data_dir=data_cfg["data_dir"],
        img_size=tuple(data_cfg["img_size"]),
        batch_size=int(train_cfg["batch_size"]),
        seed=int(config["general"]["seed"]),
        validation_source=data_cfg.get("validation_source", "folder"),
        validation_split=float(data_cfg.get("validation_split", 0.15)),
        cache_in_memory=bool(data_cfg.get("cache_in_memory", False)),
    )

    y_prob = model.predict(datasets["test"], verbose=0).ravel()
    y_true = np.concatenate([labels.numpy().ravel() for _, labels in datasets["test"]]).astype(int)
    file_paths = list_split_image_paths(Path(data_cfg["data_dir"]) / "test")

    artifact_paths = generate_evaluation_artifacts(
        y_true=y_true,
        y_prob=y_prob,
        threshold=float(eval_cfg["threshold"]),
        report_dir=eval_cfg["report_dir"],
        file_paths=file_paths,
    )

    metrics = evaluate_predictions(y_true=y_true, y_prob=y_prob, threshold=float(eval_cfg["threshold"]))["metrics"]
    print({"metrics": metrics, "artifacts": artifact_paths})


if __name__ == "__main__":
    main()
