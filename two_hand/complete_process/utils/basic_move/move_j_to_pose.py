from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complete_process.utils.base.ur_base import UR_BASE
from complete_process.utils.basic_move.process_utils import run_step, validated_pose


def main() -> None:
    parser = argparse.ArgumentParser(description="Use moveJ/IK to move the TCP to a target pose.")
    parser.add_argument("--host", default="192.168.1.5", help="UR robot IP address")
    parser.add_argument("--pose", nargs=6, type=float, required=True, metavar=("X", "Y", "Z", "RX", "RY", "RZ"))
    parser.add_argument("--speed", type=float, default=0.3, help="joint speed in rad/s")
    parser.add_argument("--acc", type=float, default=0.5, help="joint acceleration in rad/s^2")
    parser.add_argument("--timeout", type=float, default=20.0, help="motion timeout in seconds")
    args = parser.parse_args()

    pose = validated_pose(args.pose)
    with UR_BASE(args.host) as ur:
        run_step("moveJ to TCP pose", lambda: ur.move_j_ik(pose, args.speed, args.acc, args.timeout))


if __name__ == "__main__":
    main()

