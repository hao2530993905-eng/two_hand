from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complete_process.utils.base.ur_base import UR_BASE
from complete_process.utils.basic_move.process_utils import (
    offset_base_z,
    run_step,
    validated_pose,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete moveJ/moveL pick-and-place process.")
    parser.add_argument("--host", default="192.168.1.4", help="UR robot IP address")
    parser.add_argument("--gripper-port", type=int, default=63352, help="Robotiq gripper port")
    parser.add_argument("--pick-pose", nargs=6, type=float, required=True, metavar=("X", "Y", "Z", "RX", "RY", "RZ"), help="TCP pose above the object")
    parser.add_argument("--pick-distance", type=float, required=True, help="signed base-Z moveL distance; negative moves down")
    parser.add_argument("--place-pose", nargs=6, type=float, required=True, metavar=("X", "Y", "Z", "RX", "RY", "RZ"), help="TCP pose above the placement point")
    parser.add_argument("--place-distance", type=float, required=True, help="signed base-Z moveL distance; negative moves down")
    parser.add_argument("--joint-speed", type=float, default=0.3, help="moveJ speed in rad/s")
    parser.add_argument("--joint-acc", type=float, default=0.5, help="moveJ acceleration in rad/s^2")
    parser.add_argument(
        "--speed",
        "--linear-speed",
        dest="linear_speed",
        type=float,
        default=0.03,
        help="robot TCP moveL speed in m/s (default: 0.03)",
    )
    parser.add_argument("--linear-acc", type=float, default=0.1, help="moveL acceleration in m/s^2")
    parser.add_argument("--motion-timeout", type=float, default=20.0, help="timeout for each motion")
    parser.add_argument("--gripper-timeout", type=float, default=5.0, help="timeout for each gripper action")
    parser.add_argument("--gripper-speed", type=int, default=255, help="gripper speed, 0 to 255")
    parser.add_argument("--gripper-force", type=int, default=255, help="gripper force, 0 to 255")
    parser.add_argument("--retreat-after-place", action="store_true", help="move back to place pose after releasing")
    args = parser.parse_args()

    if not 0 <= args.gripper_speed <= 255:
        parser.error("--gripper-speed must be between 0 and 255")
    if not 0 <= args.gripper_force <= 255:
        parser.error("--gripper-force must be between 0 and 255")
    if args.joint_speed <= 0.0:
        parser.error("--joint-speed must be positive")
    if args.linear_speed <= 0.0:
        parser.error("--speed must be positive")
    if args.joint_acc <= 0.0:
        parser.error("--joint-acc must be positive")
    if args.linear_acc <= 0.0:
        parser.error("--linear-acc must be positive")

    pick_pose = validated_pose(args.pick_pose, "pick pose")
    place_pose = validated_pose(args.place_pose, "place pose")
    pick_target = offset_base_z(pick_pose, args.pick_distance)
    place_target = offset_base_z(place_pose, args.place_distance)

    print(f"pick approach:  {pick_pose}")
    print(f"pick target:    {pick_target}")
    print(f"place approach: {place_pose}")
    print(f"place target:   {place_target}")

    with UR_BASE(args.host, gripper_port=args.gripper_port) as ur:
        run_step(
            "open gripper",
            lambda: ur.open_gripper(
                speed=args.gripper_speed,
                force=args.gripper_force,
                timeout_s=args.gripper_timeout,
            ),
        )
        run_step("moveJ to pick pose", lambda: ur.move_j_ik(pick_pose, args.joint_speed, args.joint_acc, args.motion_timeout))
        run_step("moveL to object", lambda: ur.move_l(pick_target, args.linear_speed, args.linear_acc, args.motion_timeout))
        run_step("close gripper", lambda: ur.close_gripper(speed=args.gripper_speed, force=args.gripper_force, timeout_s=args.gripper_timeout))
        run_step(
            "moveL return to pick pose",
            lambda: ur.move_l(pick_pose, args.linear_speed, args.linear_acc, args.motion_timeout),
        )
        run_step("moveJ to place pose", lambda: ur.move_j_ik(place_pose, args.joint_speed, args.joint_acc, args.motion_timeout))
        run_step("moveL to placement point", lambda: ur.move_l(place_target, args.linear_speed, args.linear_acc, args.motion_timeout))
        run_step("open gripper", lambda: ur.open_gripper(speed=args.gripper_speed, force=args.gripper_force, timeout_s=args.gripper_timeout))
        if args.retreat_after_place:
            run_step("moveL retreat from placement point", lambda: ur.move_l(place_pose, args.linear_speed, args.linear_acc, args.motion_timeout))


if __name__ == "__main__":
    main()
