from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complete_process.utils.base.ur_base import UR_BASE


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read the current UR TCP pose without moving the robot. "
            "Position is reported in millimetres."
        )
    )
    parser.add_argument("--host", default="192.168.1.5", help="UR robot IP address")
    parser.add_argument(
        "--precision",
        type=int,
        default=6,
        help="number of decimal places in the output",
    )
    parser.add_argument(
        "--include-joints",
        action="store_true",
        help="also print current joint positions in radians and degrees",
    )
    args = parser.parse_args()
    if args.precision < 0:
        parser.error("--precision must be non-negative")

    with UR_BASE(args.host, connect_control=False) as ur:
        result = ur.get_tcp_pose()
        if not result.success or result.value is None:
            raise RuntimeError(f"failed to read TCP pose: {result.message}")

        pose = [
            round(value * 1000 if index < 3 else value, args.precision)
            for index, value in enumerate(result.value)
        ]
        print(json.dumps(pose, ensure_ascii=False))
        print("order: [x, y, z, rx, ry, rz]")
        print("units: x/y/z = mm, rx/ry/rz = rad")
        if args.include_joints:
            joints = ur.get_joint_positions()
            if not joints.success or joints.value is None:
                raise RuntimeError(f"failed to read joints: {joints.message}")
            joints_rad = [round(value, args.precision) for value in joints.value]
            joints_deg = [
                round(value * 180.0 / 3.141592653589793, args.precision)
                for value in joints.value
            ]
            print("joints_rad: " + json.dumps(joints_rad, ensure_ascii=False))
            print("joints_deg: " + json.dumps(joints_deg, ensure_ascii=False))


if __name__ == "__main__":
    main()
