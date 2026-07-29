import unittest

import numpy as np

from src.evaluate import evaluate_predictions


class EvaluatePredictionsTests(unittest.TestCase):
    def test_binary_metrics_at_configured_threshold(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.8, 0.9, 0.4])

        result = evaluate_predictions(y_true, y_prob, threshold=0.5)

        self.assertEqual(result["y_pred"].tolist(), [0, 1, 1, 0])
        self.assertAlmostEqual(result["metrics"]["accuracy"], 0.5)
        self.assertAlmostEqual(result["metrics"]["recall"], 0.5)
        self.assertAlmostEqual(result["metrics"]["specificity"], 0.5)


if __name__ == "__main__":
    unittest.main()
