"""Model design layer for transfer-learning strategy.

This module defines backbone options and assembly rules so teams can benchmark
VGG16, ResNet50V2, and MobileNetV2 under a consistent architecture and
optimization setup.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


BACKBONES = {
    "vgg16": tf.keras.applications.VGG16,
    "resnet50v2": tf.keras.applications.ResNet50V2,
    "mobilenetv2": tf.keras.applications.MobileNetV2,
}

PREPROCESSORS = {
    "vgg16": tf.keras.applications.vgg16.preprocess_input,
    "resnet50v2": tf.keras.applications.resnet_v2.preprocess_input,
    "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
}



def get_preprocessor(backbone: str):
    key = backbone.lower()
    if key not in PREPROCESSORS:
        raise ValueError(f"Unsupported backbone '{backbone}'. Use one of: {list(PREPROCESSORS)}")
    return PREPROCESSORS[key]


def build_transfer_model(
    backbone: str,
    input_shape: Tuple[int, int, int],
    learning_rate: float,
    dropout_rate: float,
    dense_units: int,
    trainable_layers: int,
    use_augmentation: bool = True,
    augmentation_layer: tf.keras.Sequential | None = None,
) -> tf.keras.Model:
    key = backbone.lower()
    if key not in BACKBONES:
        raise ValueError(f"Unsupported backbone '{backbone}'. Use one of: {list(BACKBONES)}")

    backbone_cls = BACKBONES[key]
    preprocessor = get_preprocessor(key)

    base_model = backbone_cls(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = False

    if trainable_layers > 0:
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True

    inputs = layers.Input(shape=input_shape, name="input_image")
    x = inputs

    if use_augmentation and augmentation_layer is not None:
        x = augmentation_layer(x)

    x = preprocessor(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pneumonia_probability")(x)

    model = models.Model(inputs, outputs, name=f"pneumonia_{key}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model
