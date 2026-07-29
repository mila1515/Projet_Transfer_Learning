import unittest
from unittest.mock import patch

import tensorflow as tf

from src.model import BACKBONES, PREPROCESSORS, build_transfer_model, get_preprocessor


def _tiny_backbone(*, include_top, weights, input_shape):
    del include_top, weights
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(input_shape),
            tf.keras.layers.Conv2D(2, 3, name="frozen_conv"),
            tf.keras.layers.Conv2D(4, 3, name="fine_tuned_conv"),
        ],
        name="tiny_backbone",
    )


class TransferModelTests(unittest.TestCase):
    def test_mobilenetv2_is_available(self) -> None:
        self.assertIn("mobilenetv2", BACKBONES)
        self.assertIs(
            get_preprocessor("mobilenetv2"),
            tf.keras.applications.mobilenet_v2.preprocess_input,
        )

    def test_only_requested_backbone_tail_is_trainable(self) -> None:
        with (
            patch.dict(BACKBONES, {"tiny": _tiny_backbone}),
            patch.dict(PREPROCESSORS, {"tiny": lambda inputs: inputs}),
        ):
            model = build_transfer_model(
                backbone="tiny",
                input_shape=(12, 12, 3),
                learning_rate=1e-3,
                dropout_rate=0.0,
                dense_units=4,
                trainable_layers=1,
                use_augmentation=False,
            )

        backbone = model.get_layer("tiny_backbone")
        self.assertFalse(backbone.get_layer("frozen_conv").trainable)
        self.assertTrue(backbone.get_layer("fine_tuned_conv").trainable)
        trainable_paths = [weight.path for weight in model.trainable_weights]
        self.assertTrue(any("fine_tuned_conv" in path for path in trainable_paths))
        self.assertFalse(any("frozen_conv" in path for path in trainable_paths))


if __name__ == "__main__":
    unittest.main()
