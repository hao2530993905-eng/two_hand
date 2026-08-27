from __future__ import annotations

import argparse

import _path_setup  # noqa: F401
from complete_process.utils.base.ur_base import UR_BASE


def main() -> None:
    parser = argparse.ArgumentParser(description="Read UR state without moving.")
    parser.add_argument("--host", default="192.168.1.4", help="UR robot IP address")
    args = parser.parse_args()

    with UR_BASE(args.host, connect_control=False) as ur:
        tcp = ur.get_tcp_pose()
        joints = ur.get_joint_positions()
        speed = ur.get_tcp_speed()
        print("TCP pose:", tcp)
        print("Joint positions:", joints)
        print("TCP speed:", speed)
        if not tcp.success or not joints.success:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
