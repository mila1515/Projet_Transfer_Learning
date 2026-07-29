import tempfile
import unittest
from pathlib import Path

import tensorflow as tf
from PIL import Image

from src.inference import load_model, predict_from_pil


class InferenceIntegrationTests(unittest.TestCase):
    def tearDown(self) -> None:
        load_model.cache_clear()

    def test_saved_model_is_loaded_and_used_for_prediction(self) -> None:
        inputs = tf.keras.Input((8, 8, 3))
        pooled = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        outputs = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            kernel_initializer="zeros",
            bias_initializer="zeros",
        )(pooled)
        model = tf.keras.Model(inputs, outputs)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "tiny.keras"
            model.save(model_path)

            result = predict_from_pil(
                Image.new("RGB", (12, 10), color="white"),
                model_path=str(model_path),
                img_size=(8, 8),
                threshold=0.6,
            )

        self.assertEqual(result["prediction"], "NORMAL")
        self.assertAlmostEqual(result["probability"], 0.5)
        self.assertEqual(result["threshold"], 0.6)


if __name__ == "__main__":
    unittest.main()
