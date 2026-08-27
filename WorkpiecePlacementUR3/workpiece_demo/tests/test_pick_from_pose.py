from __future__ import annotations

import math
import unittest

import _path_setup  # noqa: F401
from pick_from_pose import make_pick_pose


class PickFromPoseTests(unittest.TestCase):
    def test_make_pick_pose_moves_only_base_z(self) -> None:
        approach = [0.25, -0.1, 0.3, math.pi, 0.0, 0.5]

        pick = make_pick_pose(approach, 0.08)

        self.assertEqual(pick, [0.25, -0.1, 0.21999999999999997, math.pi, 0.0, 0.5])
        self.assertEqual(approach, [0.25, -0.1, 0.3, math.pi, 0.0, 0.5])

    def test_rejects_negative_distance(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            make_pick_pose([0.0] * 6, -0.01)

    def test_rejects_invalid_pose_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 6"):
            make_pick_pose([0.0] * 5, 0.01)


if __name__ == "__main__":
    unittest.main()
