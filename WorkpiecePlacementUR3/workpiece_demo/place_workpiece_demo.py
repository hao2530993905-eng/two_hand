from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

try:
    from complete_process.utils.base.ur_base import UR_BASE
except ModuleNotFoundError:
    from ..complete_process.utils.base.ur_base import UR_BASE


def _pose(data: dict, name: str) -> List[float]:
    values = data[name]
    if len(values) != 6:
        raise ValueError(f"{name} must contain 6 values")
    return [float(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Basic pick-and-place sequence for a workpiece.")
    parser.add_argument("--host", default="192.168.1.4", help="UR robot IP address")
    parser.add_argument("--gripper-port", type=int, default=63352, help="Robotiq gripper port")
    parser.add_argument("--config", required=True, help="JSON file with named poses")
    parser.add_argument("--speed", type=float, default=0.03, help="moveL speed in m/s")
    parser.add_argument("--acc", type=float, default=0.1, help="moveL acceleration in m/s^2")
    parser.add_argument("--timeout", type=float, default=15.0, help="timeout for each motion")
    args = parser.parse_args()

    data = json.loads(Path(args.config).read_text(encoding="utf-8"))
    home = _pose(data, "home")
    pick_approach = _pose(data, "pick_approach")
    pick = _pose(data, "pick")
    place_approach = _pose(data, "place_approach")
    place = _pose(data, "place")

    with UR_BASE(args.host, gripper_port=args.gripper_port) as ur:
        steps = [
            ("open gripper", lambda: ur.open_gripper()),
            ("move home", lambda: ur.move_l(home, args.speed, args.acc, args.timeout)),
            ("move pick approach", lambda: ur.move_l(pick_approach, args.speed, args.acc, args.timeout)),
            ("move pick", lambda: ur.move_l(pick, args.speed, args.acc, args.timeout)),
            ("close gripper", lambda: ur.close_gripper()),
            ("retreat pick", lambda: ur.move_l(pick_approach, args.speed, args.acc, args.timeout)),
            ("move place approach", lambda: ur.move_l(place_approach, args.speed, args.acc, args.timeout)),
            ("move place", lambda: ur.move_l(place, args.speed, args.acc, args.timeout)),
            ("open gripper", lambda: ur.open_gripper()),
            ("retreat place", lambda: ur.move_l(place_approach, args.speed, args.acc, args.timeout)),
            ("move home", lambda: ur.move_l(home, args.speed, args.acc, args.timeout)),
        ]

        for name, action in steps:
            result = action()
            print(name, result)
            if not result.success:
                raise RuntimeError(f"step failed: {name}: {result.message}")


if __name__ == "__main__":
    main()
