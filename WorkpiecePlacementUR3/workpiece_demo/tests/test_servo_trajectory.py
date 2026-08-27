from __future__ import annotations

import argparse

import _path_setup  # noqa: F401
from complete_process.utils.base.ur_base import UR_BASE


def interpolate(start: list[float], target: list[float], steps: int) -> list[list[float]]:
    path = []
    for index in range(steps + 1):
        ratio = index / steps
        path.append([s + (t - s) * ratio for s, t in zip(start, target)])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Follow a tiny generated servoL path and return.")
    parser.add_argument("--host", default="192.168.1.4", help="UR robot IP address")
    parser.add_argument("--dz", type=float, default=0.001, help="Z offset in meters")
    parser.add_argument("--steps", type=int, default=50, help="servo path interpolation steps")
    parser.add_argument("--dt", type=float, default=0.02, help="servo period in seconds")
    args = parser.parse_args()

    with UR_BASE(args.host) as ur:
        start = ur.get_tcp_pose_or_raise()
        target = start.copy()
        target[2] += args.dz

        out_path = interpolate(start, target, args.steps)
        back_path = interpolate(target, start, args.steps)

        out = ur.follow_servo_path(out_path, dt_s=args.dt)
        print("servo out:", out)
        if not out.success:
            raise RuntimeError(out.message)

        back = ur.follow_servo_path(back_path, dt_s=args.dt)
        print("servo back:", back)
        if not back.success:
            raise RuntimeError(back.message)


if __name__ == "__main__":
    main()
