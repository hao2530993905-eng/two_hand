from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from complete_process.utils.object_detective.depth_background_rect_detector import (
    DepthBackgroundRectangleDetector,
    height_mask,
    median_depth_background,
    white_color_mask,
)
from complete_process.utils.object_detective.rectangle_geometry import (
    detect_rectangles,
)


class DepthAlgorithmTests(unittest.TestCase):
    def test_median_background_ignores_zero_depth(self) -> None:
        frames = [
            np.array([[1000, 0]], dtype=np.uint16),
            np.array([[1010, 900]], dtype=np.uint16),
            np.array([[1020, 920]], dtype=np.uint16),
        ]
        result = median_depth_background(frames)
        np.testing.assert_array_equal(
            result, np.array([[1010, 910]], dtype=np.uint16)
        )

    def test_height_mask_keeps_only_configured_height(self) -> None:
        background = np.full((8, 8), 1000, dtype=np.uint16)
        current = background.copy()
        current[2:6, 2:6] = 970
        current[0, 0] = 800
        mask, height = height_mask(
            current, background, 20, 60, median_size=0, open_size=0, close_size=0
        )
        self.assertEqual(int(mask[3, 3]), 255)
        self.assertEqual(int(height[3, 3]), 30)
        self.assertEqual(int(mask[0, 0]), 0)
        self.assertEqual(int(mask[1, 1]), 0)

    def test_white_mask_rejects_saturated_and_dark_pixels(self) -> None:
        image = np.array(
            [[[240, 240, 240], [0, 0, 255], [120, 120, 120]]],
            dtype=np.uint8,
        )
        mask = white_color_mask(image, max_saturation=65, min_value=150)
        self.assertEqual(mask.tolist(), [[255, 0, 0]])

    def test_geometry_accepts_rectangle_and_rejects_circle(self) -> None:
        rectangle = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(rectangle, (40, 70), (160, 130), 255, cv2.FILLED)
        accepted = detect_rectangles(
            rectangle,
            min_area=100,
            max_objects=10,
            min_rectangularity=0.80,
            min_solidity=0.75,
        )
        self.assertEqual(len(accepted), 1)

        circle = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(circle, (100, 100), 50, 255, cv2.FILLED)
        rejected = detect_rectangles(
            circle,
            min_area=100,
            max_objects=10,
            min_rectangularity=0.80,
            min_solidity=0.75,
        )
        self.assertEqual(rejected, [])


class DatasetSavingTests(unittest.TestCase):
    def make_detector(self, dataset_dir: Path) -> DepthBackgroundRectangleDetector:
        detector = DepthBackgroundRectangleDetector.__new__(
            DepthBackgroundRectangleDetector
        )
        detector.collect_dataset = True
        detector.dataset_dir = dataset_dir
        detector.dataset_split = "train"
        detector.class_id = 0
        detector.saved_samples = []
        return detector

    def test_positive_sample_writes_image_label_preview_and_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = Path(directory) / "white_rectangle_dataset"
            detector = self.make_detector(dataset_dir)
            detector.initialize_dataset()
            image = np.full((100, 200, 3), 220, dtype=np.uint8)
            objects = [{
                "corners_px": [[20, 20], [180, 20], [180, 80], [20, 80]]
            }]

            detector.save_dataset_sample(image, objects)

            self.assertEqual(len(detector.saved_samples), 1)
            image_path, label_path, preview_path = detector.saved_samples[0]
            self.assertTrue(image_path.is_file())
            self.assertTrue(preview_path.is_file())
            fields = label_path.read_text(encoding="utf-8").split()
            self.assertEqual(len(fields), 9)
            self.assertEqual(fields[0], "0")
            self.assertIn("0: white_rectangle", (
                dataset_dir / "dataset.yaml"
            ).read_text(encoding="utf-8"))

            detector.undo_last_sample()
            self.assertFalse(image_path.exists())
            self.assertFalse(label_path.exists())
            self.assertFalse(preview_path.exists())

    def test_negative_sample_has_empty_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            detector = self.make_detector(Path(directory) / "dataset")
            detector.initialize_dataset()
            detector.save_dataset_sample(
                np.zeros((40, 60, 3), dtype=np.uint8), [], negative=True
            )
            _, label_path, _ = detector.saved_samples[0]
            self.assertEqual(label_path.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
