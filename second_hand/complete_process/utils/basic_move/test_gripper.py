from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complete_process.utils.base.ur_base import UR_BASE


def main() -> None:
    parser = argparse.ArgumentParser(description="Open or close a Robotiq gripper.")
    parser.add_argument("--host", default="192.168.1.5", help="UR robot IP address")
    parser.add_argument("--port", type=int, default=63352, help="Robotiq gripper port")
    parser.add_argument("--action", choices=["open", "close"], required=True)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--force", type=int, default=255, help="gripper force, 0 (minimum) to 255 (maximum)")
    parser.add_argument("--monitor", action="store_true", help="continuously read force setting, position and object status")
    parser.add_argument("--interval", type=float, default=0.2, help="monitor output interval in seconds")
    args = parser.parse_args()
    if not 0 <= args.force <= 255:
        parser.error("--force must be between 0 and 255")
    if args.interval <= 0.0:
        parser.error("--interval must be positive")

    with UR_BASE(args.host, gripper_port=args.port, connect_control=False) as ur:
        if args.action == "open":
            result = ur.open_gripper(force=args.force, timeout_s=args.timeout)
        else:
            result = ur.close_gripper(force=args.force, timeout_s=args.timeout)
        print(result)
        if not result.success:
            raise SystemExit(1)
        if args.monitor:
            if ur.gripper is None:
                raise RuntimeError("gripper is not connected")
            print("Monitoring registers. FOR is a 0-255 setting, not measured force in newtons. Press Ctrl+C to stop.")
            try:
                while True:
                    ok_force, force, force_message = ur.gripper.get_force_setting()
                    ok_pos, position, position_message = ur.gripper.get_position()
                    ok_obj, obj, obj_message = ur.gripper.get_object_status()
                    if not ok_force:
                        raise RuntimeError(force_message)
                    if not ok_pos:
                        raise RuntimeError(position_message)
                    if not ok_obj:
                        raise RuntimeError(obj_message)
                    print(
                        f"FOR={force:3d}/255  POS={position:3d}/255  "
                        f"OBJ={obj} ({object_status_text(obj)})",
                        flush=True,
                    )
                    time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nMonitoring stopped.")


def object_status_text(value: int) -> str:
    return {
        0: "moving/no object",
        1: "object detected while opening",
        2: "object detected while closing",
        3: "requested position reached",
    }.get(value, "unknown")


if __name__ == "__main__":
    main()

