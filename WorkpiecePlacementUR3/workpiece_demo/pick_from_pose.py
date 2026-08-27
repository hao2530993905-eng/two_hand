from __future__ import annotations

import argparse
import math
from typing import Iterable, List

try:
    from complete_process.utils.base.ur_base import UR_BASE
except ModuleNotFoundError:
    from ..complete_process.utils.base.ur_base import UR_BASE


def make_pick_pose(approach_pose: Iterable[float], down_distance: float) -> List[float]:
    """Return a pose translated down along the robot base Z axis."""
    pose = [float(value) for value in approach_pose]
    if len(pose) != 6:
        raise ValueError("pose must contain exactly 6 values: x y z rx ry rz")
    if not all(math.isfinite(value) for value in pose):
        raise ValueError("pose contains a non-finite value")

    distance = float(down_distance)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("down distance must be a non-negative finite value")

    pick_pose = pose.copy()
    pick_pose[2] -= distance
    return pick_pose


def _run_step(name: str, action: object) -> None:
    result = action()
    print(f"{name}: {result}")
    if not result.success:
        raise RuntimeError(f"step failed: {name}: {result.message}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Move to an approach TCP pose, descend along the base -Z axis, "
            "and close the Robotiq gripper. Pose units are metres and radians."
        )
    )
    parser.add_argument("--host", default="192.168.1.4", help="UR robot IP address")
    parser.add_argument("--gripper-port", type=int, default=63352, help="Robotiq gripper port")
    parser.add_argument(
        "--pose",
        nargs=6,
        type=float,
        required=True,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="approach TCP pose in metres and radians",
    )
    parser.add_argument("--down", type=float, required=True, help="descent distance in metres")
    parser.add_argument("--speed", type=float, default=0.03, help="moveL speed in m/s")
    parser.add_argument("--acc", type=float, default=0.1, help="moveL acceleration in m/s^2")
    parser.add_argument("--motion-timeout", type=float, default=15.0, help="timeout for each motion in seconds")
    parser.add_argument("--gripper-timeout", type=float, default=5.0, help="timeout for each gripper action")
    parser.add_argument("--gripper-speed", type=int, default=255, help="gripper speed, 0 to 255")
    parser.add_argument("--gripper-force", type=int, default=255, help="gripper force, 0 to 255")
    parser.add_argument(
        "--lift-after",
        action="store_true",
        help="return to the approach pose after closing the gripper",
    )
    args = parser.parse_args()

    approach_pose = [float(value) for value in args.pose]
    pick_pose = make_pick_pose(approach_pose, args.down)
    if not 0 <= args.gripper_speed <= 255:
        parser.error("--gripper-speed must be between 0 and 255")
    if not 0 <= args.gripper_force <= 255:
        parser.error("--gripper-force must be between 0 and 255")
    if args.speed <= 0.0 or args.acc <= 0.0:
        parser.error("--speed and --acc must be positive")

    print(f"approach pose: {approach_pose}")
    print(f"pick pose:     {pick_pose}")

    with UR_BASE(args.host, gripper_port=args.gripper_port) as ur:
        _run_step(
            "open gripper",
            lambda: ur.open_gripper(
                speed=args.gripper_speed,
                force=args.gripper_force,
                timeout_s=args.gripper_timeout,
            ),
        )
        _run_step(
            "move to approach pose",
            lambda: ur.move_l(approach_pose, args.speed, args.acc, args.motion_timeout),
        )
        _run_step(
            "move down to pick pose",
            lambda: ur.move_l(pick_pose, args.speed, args.acc, args.motion_timeout),
        )
        _run_step(
            "close gripper",
            lambda: ur.close_gripper(
                speed=args.gripper_speed,
                force=args.gripper_force,
                timeout_s=args.gripper_timeout,
            ),
        )
        if args.lift_after:
            _run_step(
                "lift to approach pose",
                lambda: ur.move_l(approach_pose, args.speed, args.acc, args.motion_timeout),
            )


if __name__ == "__main__":
    main()
