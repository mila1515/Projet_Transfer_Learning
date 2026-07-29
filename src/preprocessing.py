"""Preprocessing standards for data quality and model consistency.

This module defines normalization and augmentation steps that reduce input
variance and improve robustness during both training and production scoring.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def build_augmentation_pipeline() -> tf.keras.Sequential:
    """Production-oriented augmentation for chest X-ray classification."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        ],
        name="augmentation_pipeline",
    )
