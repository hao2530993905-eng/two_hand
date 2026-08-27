from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complete_process import two_hand_pick_and_place as flow


class TwoHandConfigurationTests(unittest.TestCase):
    def test_command_defaults_match_verified_commands(self):
        self.assertEqual(flow.DEFAULT_RECEIVER_TARGET_POSE_MM[:3], (
            -187.325487, -290.96447, 339.416991
        ))
        self.assertEqual(flow.DEFAULT_SUPPLIER_FINAL_POSE[:3], (
            -0.323390523, 0.120168609, 0.364210170
        ))

    def test_tool_z_translation_preserves_rotation(self):
        pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]
        moved = flow.tool_z_pose(pose, 0.045)
        np.testing.assert_allclose(moved, [0.1, 0.2, 0.345, 0.0, 0.0, 0.0])

    def test_runtime_sources_do_not_reference_old_projects(self):
        forbidden = ("/home/sjh/second_hand", "/home/sjh/WorkpiecePlacementUR3")
        for path in (PROJECT_ROOT / "complete_process").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, source, str(path))

    def test_handoff_timers_bracket_release_and_receiver_retract(self):
        events = []
        lock = threading.Lock()

        class Robot:
            def __init__(self, name):
                self.name = name

            def record(self, action):
                with lock:
                    events.append("{}:{}".format(self.name, action))
                return SimpleNamespace(success=True, message="ok")

            def move_j_ik(self, *unused):
                return self.record("move_j_ik")

            def move_j(self, *unused):
                return self.record("move_j")

            def move_l(self, *unused):
                return self.record("move_l")

            def open_gripper(self, **unused):
                return self.record("open")

            def close_gripper(self, **unused):
                return self.record("close")

            def get_tcp_pose_or_raise(self):
                return [0.0, 0.2, 0.2, 0.0, 0.0, 0.0]

        args = SimpleNamespace(
            movej_speed=0.3, movej_acc=0.3, movel_speed=0.3, movel_acc=0.3,
            second_hand_motion_timeout=100.0, gripper_timeout=5.0,
            gripper_speed=255, gripper_force=255,
            joint_speed=0.3, joint_acc=0.3,
            linear_speed=0.3, linear_acc=0.3, motion_timeout=60.0,
            lift_distance=0.13, post_pick_lift=0.15,
            second_hand_workspace=(-0.45, 0.45, 0.05, 0.75, 0.03, 0.65),
            observation_joints=None, put_second_hand_time=0.4,
            wait_second_hand=0.6,
        )
        supplier = {
            "pregrasp": [0.0] * 6, "grasp": [0.0] * 6,
            "lift": [0.0] * 6, "handoff": [0.0] * 6,
        }
        receiver = {
            "approach": [0.0] * 6, "grasp": [0.0] * 6,
            "lifted": [0.0] * 6, "place": [0.0] * 6,
            "release": [0.0] * 6, "slot": 1,
        }

        def fake_sleep(seconds):
            with lock:
                events.append("sleep:{:.1f}".format(seconds))

        with patch.object(flow.time, "sleep", side_effect=fake_sleep):
            flow.execute_sequence(args, Robot("supplier"), Robot("receiver"),
                                  supplier, receiver)

        receiver_close = events.index("receiver:close")
        put_wait = events.index("sleep:0.4")
        supplier_release = events.index("supplier:open")
        receiver_wait = events.index("sleep:0.6")
        receiver_retract = events.index("receiver:move_l", receiver_close)
        self.assertLess(receiver_close, put_wait)
        self.assertLess(put_wait, supplier_release)
        self.assertLess(supplier_release, receiver_wait)
        self.assertLess(receiver_wait, receiver_retract)


if __name__ == "__main__":
    unittest.main()
