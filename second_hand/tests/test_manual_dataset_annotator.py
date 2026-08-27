from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from complete_process.utils.object_detective.manual_dataset_annotator import (
    DatasetWriter,
    annotation_corners,
    normalized_box,
    rotated_rectangle_corners,
    yolo_detect_line,
    yolo_obb_line,
)


class ManualLabelTests(unittest.TestCase):
    def test_normalized_box_orders_and_clamps_coordinates(self) -> None:
        self.assertEqual(
            normalized_box((90, 70, -5, 10), width=80, height=60),
            (0, 10, 79, 59),
        )

    def test_obb_label_has_class_and_eight_coordinates(self) -> None:
        fields = yolo_obb_line(
            0, (20, 10, 80, 50), width=100, height=100
        ).split()
        self.assertEqual(len(fields), 9)
        self.assertEqual(fields[0], "0")
        self.assertEqual(
            [float(value) for value in fields[1:]],
            [0.2, 0.1, 0.8, 0.1, 0.8, 0.5, 0.2, 0.5],
        )

    def test_standard_detect_label_has_xywh(self) -> None:
        self.assertEqual(
            yolo_detect_line(0, (20, 10, 80, 50), 100, 100),
            "0 0.500000 0.300000 0.600000 0.400000",
        )

    def test_rotated_centerline_and_width_produce_four_corners(self) -> None:
        corners = rotated_rectangle_corners(
            (20, 50), (80, 50), (50, 40)
        )
        np.testing.assert_allclose(
            corners,
            np.array(
                [[20, 40], [80, 40], [80, 60], [20, 60]],
                dtype=np.float32,
            ),
        )
        checked = annotation_corners(corners, width=100, height=100)
        self.assertGreater(abs(float(cv2.contourArea(checked))), 1000.0)

    def test_rotated_obb_label_preserves_diagonal_corners(self) -> None:
        corners = rotated_rectangle_corners(
            (20, 20), (80, 80), (40, 60)
        )
        fields = yolo_obb_line(0, corners, 100, 100).split()
        self.assertEqual(len(fields), 9)
        points = np.asarray(
            [float(value) for value in fields[1:]], dtype=np.float32
        ).reshape(4, 2)
        self.assertFalse(np.allclose(points[0, 1], points[1, 1]))


class ManualDatasetWriterTests(unittest.TestCase):
    def test_save_positive_negative_and_undo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = DatasetWriter(
                Path(directory) / "manual_dataset",
                split="train",
                label_format="obb",
            )
            image = np.full((100, 200, 3), 180, dtype=np.uint8)

            positive_path, count = writer.save(
                image, [(20, 20, 180, 80)]
            )
            self.assertEqual(count, 1)
            positive_label = (
                writer.dataset_dir / "labels" / "train"
                / positive_path.with_suffix(".txt").name
            )
            self.assertEqual(
                len(positive_label.read_text(encoding="utf-8").split()), 9
            )

            negative_path, count = writer.save(image, [], negative=True)
            self.assertEqual(count, 0)
            negative_label = (
                writer.dataset_dir / "labels" / "train"
                / negative_path.with_suffix(".txt").name
            )
            self.assertEqual(
                negative_label.read_text(encoding="utf-8"), ""
            )

            removed = writer.undo()
            self.assertEqual(removed, negative_path)
            self.assertFalse(negative_path.exists())
            self.assertFalse(negative_label.exists())

            yaml_text = (
                writer.dataset_dir / "dataset.yaml"
            ).read_text(encoding="utf-8")
            self.assertIn("0: white_rectangle", yaml_text)

    def test_positive_requires_a_box(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = DatasetWriter(Path(directory), split="val")
            with self.assertRaises(ValueError):
                writer.save(np.zeros((20, 20, 3), dtype=np.uint8), [])


if __name__ == "__main__":
    unittest.main()
