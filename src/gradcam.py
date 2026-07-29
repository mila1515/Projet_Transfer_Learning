"""Explainability utilities for clinical and stakeholder transparency.

This module generates Grad-CAM heatmaps to show where the model focuses,
supporting interpretability reviews and improving trust in prediction outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image


def _find_last_conv_layer_name(model: tf.keras.Model) -> str:
    """Find the last top-level layer producing a spatial feature map.

    Application backbones are nested Keras models. Their top-level output is
    connected to the classifier graph and is therefore a reliable Grad-CAM
    target, unlike an internal layer tensor from the nested graph.
    """
    for layer in reversed(model.layers):
        output = getattr(layer, "output", None)
        shape = getattr(output, "shape", None)
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError("No 4D convolution-like layer found for Grad-CAM.")


def _get_layer_recursive(model: tf.keras.Model, layer_name: str):
    try:
        layer = model.get_layer(layer_name)
    except ValueError as exc:
        raise ValueError(
            f"Grad-CAM layer '{layer_name}' must be a top-level model layer."
        ) from exc

    shape = getattr(getattr(layer, "output", None), "shape", None)
    if shape is None or len(shape) != 4:
        raise ValueError(f"Grad-CAM layer '{layer_name}' must have a 4D output.")
    return layer


def _build_grad_model(model: tf.keras.Model, target_layer_name: str) -> tf.keras.Model:
    """Reconnect a linear top-level graph, including after Keras deserialization."""
    _get_layer_recursive(model, target_layer_name)
    x = model.inputs[0]
    target_output = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        x = layer(x)
        if layer.name == target_layer_name:
            target_output = x

    if target_output is None:
        raise ValueError(f"Grad-CAM layer '{target_layer_name}' is not connected.")
    return tf.keras.Model(model.inputs, [target_output, x])


def make_gradcam_heatmap(
    image_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: Optional[str] = None,
) -> np.ndarray:
    if len(model.inputs) != 1:
        raise ValueError("Grad-CAM currently supports single-input models only.")

    if last_conv_layer_name is None:
        last_conv_layer_name = _find_last_conv_layer_name(model)

    grad_model = _build_grad_model(model, last_conv_layer_name)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([image_array], training=False)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.math.divide_no_nan(tf.maximum(heatmap, 0), tf.math.reduce_max(heatmap))
    return heatmap.numpy()


def save_gradcam_overlay(
    image_path: str | Path,
    heatmap: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.4,
) -> str:
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image_np.shape[1], image_np.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    superimposed = jet_heatmap * alpha + image_np
    superimposed = np.uint8(np.clip(superimposed, 0, 255))
    superimposed = cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), superimposed)

    return str(output)
