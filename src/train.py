"""Training lifecycle orchestration for production-ready model delivery.

This module runs the full experiment cycle, from data ingestion to MLflow
tracking and post-training evaluation, to support repeatable model releases.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import mlflow
import mlflow.keras
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src.config import load_config
from src.data_loader import (
    build_datasets,
    compute_class_weights,
    get_data_augmentation,
    list_split_image_paths,
)
from src.evaluate import evaluate_predictions, generate_evaluation_artifacts
from src.model import build_transfer_model
from src.utils import ensure_dir, set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pneumonia classifier with transfer learning.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file.")
    parser.add_argument("--backbone", type=str, default=None, help="Override model backbone from config.")
    parser.add_argument("--experiment", type=str, default=None, help="Override MLflow experiment name.")
    return parser.parse_args()


def train(config: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(config["general"]["seed"])
    set_seed(seed)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    eval_cfg = config["evaluation"]
    mlflow_cfg = config["mlflow"]

    img_size = tuple(data_cfg["img_size"])
    input_shape = (img_size[0], img_size[1], 3)

    datasets = build_datasets(
        data_dir=data_cfg["data_dir"],
        img_size=img_size,
        batch_size=int(train_cfg["batch_size"]),
        seed=seed,
        validation_source=data_cfg.get("validation_source", "folder"),
        validation_split=float(data_cfg.get("validation_split", 0.15)),
        cache_in_memory=bool(data_cfg.get("cache_in_memory", False)),
    )

    augmentation = get_data_augmentation()
    model = build_transfer_model(
        backbone=model_cfg["backbone"],
        input_shape=input_shape,
        learning_rate=float(train_cfg["learning_rate"]),
        dropout_rate=float(model_cfg["dropout_rate"]),
        dense_units=int(model_cfg["dense_units"]),
        trainable_layers=int(model_cfg["trainable_layers"]),
        use_augmentation=bool(model_cfg["use_augmentation"]),
        augmentation_layer=augmentation,
    )

    model_dir = ensure_dir(train_cfg["model_dir"])
    report_dir = ensure_dir(eval_cfg["report_dir"])

    best_model_path = model_dir / f"best_{model_cfg['backbone'].lower()}.keras"
    final_model_path = model_dir / train_cfg["model_output_name"]

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=int(train_cfg["patience"]), restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7),
        ModelCheckpoint(best_model_path, monitor="val_loss", save_best_only=True),
    ]

    class_weights = None
    if train_cfg.get("use_class_weights", True):
        class_weights = compute_class_weights(Path(data_cfg["data_dir"]) / "train")

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=f"{model_cfg['backbone']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        mlflow.set_tag("project", "pneumonia-transfer-learning")
        mlflow.set_tag("stage", "training")
        mlflow.set_tag("model_version", datetime.now().strftime("%Y%m%d%H%M%S"))

        mlflow.log_params(
            {
                "seed": seed,
                "backbone": model_cfg["backbone"],
                "img_size": f"{img_size[0]}x{img_size[1]}",
                "batch_size": train_cfg["batch_size"],
                "epochs": train_cfg["epochs"],
                "learning_rate": train_cfg["learning_rate"],
                "dropout_rate": model_cfg["dropout_rate"],
                "trainable_layers": model_cfg["trainable_layers"],
                "threshold": eval_cfg["threshold"],
                "use_class_weights": class_weights is not None,
            }
        )
        if class_weights is not None:
            mlflow.log_params(
                {
                    "class_weight_normal": class_weights[0],
                    "class_weight_pneumonia": class_weights[1],
                }
            )

        history = model.fit(
            datasets["train"],
            validation_data=datasets["val"],
            epochs=int(train_cfg["epochs"]),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

        model.save(final_model_path)

        y_prob = model.predict(datasets["test"], verbose=0).ravel()
        y_true = np.concatenate([labels.numpy().ravel() for _, labels in datasets["test"]]).astype(int)
        threshold = float(eval_cfg["threshold"])
        file_paths = list_split_image_paths(Path(data_cfg["data_dir"]) / "test")
        evaluation_result = evaluate_predictions(y_true=y_true, y_prob=y_prob, threshold=threshold)
        metrics = evaluation_result["metrics"]
        generate_evaluation_artifacts(
            y_true=y_true,
            y_prob=y_prob,
            threshold=threshold,
            report_dir=report_dir,
            file_paths=list(file_paths),
        )

        for key, values in history.history.items():
            mlflow.log_metric(f"train_{key}_last", float(values[-1]))

        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(report_dir), artifact_path="reports")
        mlflow.keras.log_model(model, name="model")

        if mlflow_cfg.get("register_model", False) and mlflow_cfg.get("registered_model_name"):
            model_uri = f"runs:/{run.info.run_id}/model"
            try:
                mlflow.register_model(model_uri=model_uri, name=mlflow_cfg["registered_model_name"])
            except Exception as exc:
                print(f"Model registration failed: {exc}")

        return {
            "run_id": run.info.run_id,
            "model_path": str(final_model_path),
            "best_model_path": str(best_model_path),
            "metrics": metrics,
        }


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    if args.backbone:
        config["model"]["backbone"] = args.backbone
    if args.experiment:
        config["mlflow"]["experiment_name"] = args.experiment

    results = train(config)
    print("Training complete")
    print(results)


if __name__ == "__main__":
    main()
