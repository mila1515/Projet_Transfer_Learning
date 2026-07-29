import tempfile
import unittest
from pathlib import Path

from src.data_loader import compute_class_weights, list_split_image_paths


class SplitImagePathsTests(unittest.TestCase):
    def test_paths_follow_class_then_filename_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            normal = root / "NORMAL"
            pneumonia = root / "PNEUMONIA"
            normal.mkdir()
            pneumonia.mkdir()
            (normal / "b.jpeg").touch()
            (normal / "a.jpeg").touch()
            (pneumonia / "c.png").touch()
            (pneumonia / "ignored.txt").touch()

            paths = list_split_image_paths(root)

            self.assertEqual(
                [Path(path).name for path in paths],
                ["a.jpeg", "b.jpeg", "c.png"],
            )

    def test_class_weights_compensate_imbalance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            normal = root / "NORMAL"
            pneumonia = root / "PNEUMONIA"
            normal.mkdir()
            pneumonia.mkdir()
            (normal / "normal.jpeg").touch()
            for index in range(3):
                (pneumonia / f"pneumonia_{index}.jpeg").touch()

            weights = compute_class_weights(root)

            self.assertEqual(weights, {0: 2.0, 1: 2.0 / 3.0})


if __name__ == "__main__":
    unittest.main()
