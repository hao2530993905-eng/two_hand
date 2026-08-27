from __future__ import annotations

import math
import unittest

from complete_process.utils.basic_move.process_utils import offset_base_z, validated_pose


class ProcessUtilsTests(unittest.TestCase):
    def test_validated_pose_returns_float_copy(self) -> None:
        values = [1, 2, 3, 4, 5, 6]
        self.assertEqual(validated_pose(values), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_validated_pose_rejects_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            validated_pose([0.0] * 5)

    def test_validated_pose_rejects_non_finite_value(self) -> None:
        with self.assertRaises(ValueError):
            validated_pose([0.0, 0.0, math.nan, 0.0, 0.0, 0.0])

    def test_offset_base_z_does_not_mutate_input(self) -> None:
        pose = [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]
        target = offset_base_z(pose, -0.05)
        self.assertEqual(pose, [0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(target[2], 0.25)


if __name__ == "__main__":
    unittest.main()
