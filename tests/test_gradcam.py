import tempfile
import unittest
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.gradcam import _find_last_conv_layer_name, make_gradcam_heatmap


class GradCamTests(unittest.TestCase):
    @staticmethod
    def _nested_model() -> tf.keras.Model:
        backbone = tf.keras.Sequential(
            [
                tf.keras.layers.Input((16, 16, 3)),
                tf.keras.layers.Conv2D(4, 3, activation="relu"),
            ],
            name="backbone",
        )
        inputs = tf.keras.Input((16, 16, 3))
        features = backbone(inputs)
        pooled = tf.keras.layers.GlobalAveragePooling2D()(features)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(pooled)
        return tf.keras.Model(inputs, outputs)

    def test_finds_connected_nested_backbone_output(self) -> None:
        model = self._nested_model()
        self.assertEqual(_find_last_conv_layer_name(model), "backbone")

    def test_heatmap_is_finite_and_spatial(self) -> None:
        model = self._nested_model()
        image = np.ones((1, 16, 16, 3), dtype=np.float32)

        heatmap = make_gradcam_heatmap(image, model)

        self.assertEqual(heatmap.shape, (14, 14))
        self.assertTrue(np.isfinite(heatmap).all())
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_heatmap_works_after_model_reload(self) -> None:
        model = self._nested_model()
        image = np.ones((1, 16, 16, 3), dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "gradcam.keras"
            model.save(model_path)
            reloaded = tf.keras.models.load_model(model_path)
            heatmap = make_gradcam_heatmap(image, reloaded)

        self.assertEqual(heatmap.shape, (14, 14))
        self.assertTrue(np.isfinite(heatmap).all())


if __name__ == "__main__":
    unittest.main()
