from __future__ import annotations

import argparse

try:
    from complete_process.utils.base.ur_base import UR_BASE
except ModuleNotFoundError:
    from ..complete_process.utils.base.ur_base import UR_BASE


def main() -> None:
    parser = argparse.ArgumentParser(description="Move the UR robot to six joint positions in radians.")
    parser.add_argument("joints", nargs=6, type=float, metavar="JOINT")
    parser.add_argument("--host", default="192.168.1.4", help="UR robot IP address")
    parser.add_argument("--speed", type=float, default=0.1, help="joint speed in rad/s")
    parser.add_argument("--acc", type=float, default=0.2, help="joint acceleration in rad/s^2")
    parser.add_argument("--timeout", type=float, default=30.0, help="motion timeout in seconds")
    args = parser.parse_args()

    with UR_BASE(args.host) as robot:
        print("Control backend:", robot.control_backend)
        result = robot.move_j(
            args.joints,
            speed=args.speed,
            acceleration=args.acc,
            timeout_s=args.timeout,
        )
        print(result)
        if not result.success:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
