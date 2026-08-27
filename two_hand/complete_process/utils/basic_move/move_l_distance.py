from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complete_process.utils.base.ur_base import UR_BASE
from complete_process.utils.basic_move.process_utils import offset_base_z, run_step


def main() -> None:
    parser = argparse.ArgumentParser(description="MoveL by a signed distance along the robot base Z axis.")
    parser.add_argument("--host", default="192.168.1.5", help="UR robot IP address")
    parser.add_argument("--distance", type=float, required=True, help="signed Z distance in metres; negative moves down")
    parser.add_argument("--speed", type=float, default=0.03, help="TCP speed in m/s")
    parser.add_argument("--acc", type=float, default=0.1, help="TCP acceleration in m/s^2")
    parser.add_argument("--timeout", type=float, default=15.0, help="motion timeout in seconds")
    args = parser.parse_args()

    with UR_BASE(args.host) as ur:
        start = ur.get_tcp_pose_or_raise()
        target = offset_base_z(start, args.distance)
        print(f"start pose:  {start}")
        print(f"target pose: {target}")
        run_step("moveL Z distance", lambda: ur.move_l(target, args.speed, args.acc, args.timeout))


if __name__ == "__main__":
    main()

