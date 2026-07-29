"""Data ingestion layer for model development and monitoring.

This module structures train, validation, and test inputs so performance
metrics reflect real split behavior and support reliable model comparison
across experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import tensorflow as tf

from src.preprocessing import build_augmentation_pipeline


AUTOTUNE = tf.data.AUTOTUNE
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png"}


def get_data_augmentation() -> tf.keras.Sequential:
    """Data augmentation pipeline for chest X-ray images."""
    return build_augmentation_pipeline()


def _make_dataset(
    split_dir: Path,
    img_size: Tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    cache_in_memory: bool,
) -> tf.keras.utils.image_dataset_from_directory:
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split folder not found: {split_dir}")

    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        class_names=CLASS_NAMES,
        label_mode="binary",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )

    if cache_in_memory:
        ds = ds.cache()
    return ds.prefetch(AUTOTUNE)


def _make_train_validation_datasets(
    train_dir: Path,
    img_size: Tuple[int, int],
    batch_size: int,
    seed: int,
    validation_split: float,
    cache_in_memory: bool,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create disjoint, reproducible train/validation subsets."""
    if not train_dir.exists():
        raise FileNotFoundError(f"Dataset split folder not found: {train_dir}")
    if not 0.0 < validation_split < 1.0:
        raise ValueError("validation_split must be strictly between 0 and 1.")

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        class_names=CLASS_NAMES,
        label_mode="binary",
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="both",
    )
    if cache_in_memory:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()
    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE)


def list_split_image_paths(split_dir: str | Path) -> list[str]:
    """List image paths in the same class/file order used for unshuffled loading."""
    root = Path(split_dir)
    paths: list[str] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        paths.extend(
            str(path)
            for path in sorted(class_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return paths


def compute_class_weights(train_dir: str | Path) -> Dict[int, float]:
    """Compute balanced weights using the explicit model class order."""
    root = Path(train_dir)
    counts = {
        index: sum(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in (root / class_name).rglob("*")
        )
        for index, class_name in enumerate(CLASS_NAMES)
    }
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Both classes must contain images; observed counts: {counts}")

    total = sum(counts.values())
    return {
        index: total / (len(CLASS_NAMES) * count)
        for index, count in counts.items()
    }


def build_datasets(
    data_dir: str | Path,
    img_size: Tuple[int, int],
    batch_size: int,
    seed: int,
    validation_source: str = "folder",
    validation_split: float = 0.15,
    cache_in_memory: bool = False,
) -> Dict[str, tf.data.Dataset]:
    """Build train/val/test datasets from Chest X-ray folder structure."""
    root = Path(data_dir)

    if validation_source == "train_split":
        train_ds, val_ds = _make_train_validation_datasets(
            root / "train",
            img_size,
            batch_size,
            seed,
            validation_split,
            cache_in_memory,
        )
    elif validation_source == "folder":
        train_ds = _make_dataset(
            root / "train", img_size, batch_size, shuffle=True, seed=seed,
            cache_in_memory=cache_in_memory,
        )
        val_ds = _make_dataset(
            root / "val", img_size, batch_size, shuffle=False, seed=seed,
            cache_in_memory=cache_in_memory,
        )
    else:
        raise ValueError("validation_source must be either 'train_split' or 'folder'.")
    test_ds = _make_dataset(
        root / "test", img_size, batch_size, shuffle=False, seed=seed,
        cache_in_memory=cache_in_memory,
    )

    return {"train": train_ds, "val": val_ds, "test": test_ds}
