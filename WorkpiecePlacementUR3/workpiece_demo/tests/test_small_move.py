from __future__ import annotations

import argparse

import _path_setup  # noqa: F401
from complete_process.utils.base.ur_base import UR_BASE


def main() -> None:
    parser = argparse.ArgumentParser(description="Move TCP by a tiny offset and return.")
    parser.add_argument("--host", default="192.168.1.4", help="UR robot IP address")
    parser.add_argument("--dx", type=float, default=0.0, help="X offset in meters")
    parser.add_argument("--dy", type=float, default=0.0, help="Y offset in meters")
    parser.add_argument("--dz", type=float, default=0.001, help="Z offset in meters")
    parser.add_argument("--speed", type=float, default=0.01, help="moveL speed in m/s")
    parser.add_argument("--acc", type=float, default=0.03, help="moveL acceleration in m/s^2")
    parser.add_argument("--timeout", type=float, default=10.0, help="timeout in seconds")
    args = parser.parse_args()

    with UR_BASE(args.host) as ur:
        start = ur.get_tcp_pose_or_raise()
        target = start.copy()
        target[0] += args.dx
        target[1] += args.dy
        target[2] += args.dz

        print("start:", start)
        print("target:", target)
        out = ur.move_l(target, speed=args.speed, acceleration=args.acc, timeout_s=args.timeout)
        print("move out:", out)
        if not out.success:
            raise RuntimeError(out.message)

        back = ur.move_l(start, speed=args.speed, acceleration=args.acc, timeout_s=args.timeout)
        print("move back:", back)
        if not back.success:
            raise RuntimeError(back.message)


if __name__ == "__main__":
    main()
