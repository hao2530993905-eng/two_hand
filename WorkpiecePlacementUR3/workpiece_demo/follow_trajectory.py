from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

try:
    from complete_process.utils.base.ur_base import UR_BASE
except ModuleNotFoundError:
    from ..complete_process.utils.base.ur_base import UR_BASE


def load_trajectory(path: Path) -> List[List[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        points = data.get("poses")
    else:
        points = data
    if not isinstance(points, list):
        raise ValueError("trajectory JSON must be a list or an object with a 'poses' list")
    return [[float(v) for v in pose] for pose in points]


def follow_trajectory(
    host: str,
    trajectory: Iterable[Iterable[float]],
    dt_s: float,
    lookahead_time: float,
    gain: float,
    timeout_s: float,
) -> bool:
    with UR_BASE(host) as ur:
        result = ur.follow_servo_path(
            trajectory,
            dt_s=dt_s,
            lookahead_time=lookahead_time,
            gain=gain,
            path_timeout_s=timeout_s,
        )
        print(result)
        return result.success


def main() -> None:
    parser = argparse.ArgumentParser(description="Follow a TCP pose trajectory with UR servoL.")
    parser.add_argument("--host", default="192.168.1.4", help="UR robot IP address")
    parser.add_argument("--trajectory", required=True, help="JSON file containing poses")
    parser.add_argument("--dt", type=float, default=0.02, help="servoL control period in seconds")
    parser.add_argument("--lookahead", type=float, default=0.1, help="servoL lookahead time")
    parser.add_argument("--gain", type=float, default=300.0, help="servoL gain")
    parser.add_argument("--timeout", type=float, default=30.0, help="overall path timeout in seconds")
    args = parser.parse_args()

    trajectory = load_trajectory(Path(args.trajectory))
    ok = follow_trajectory(args.host, trajectory, args.dt, args.lookahead, args.gain, args.timeout)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
