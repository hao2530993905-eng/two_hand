#!/usr/bin/env python3
"""Print the ArUco pose in camera/base frames and the current UR3 TCP pose."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import rospy
import tf2_ros
import yaml
from tf.transformations import (
    euler_from_matrix,
    quaternion_from_matrix,
    quaternion_matrix,
)


CALIBRATION_YAML = (
    Path.home()
    / ".ros/easy_handeye/ur3_d455_handeye_eye_on_base.yaml"
)
FALLBACK_CALIBRATION_TRANSLATION = np.array(
    [-0.01044787244394433, 0.4078309180113589, 0.8614295017631832],
    dtype=np.float64,
)
FALLBACK_CALIBRATION_QUATERNION = np.array(
    [-0.49690966616336485, 0.512592084317204,
     0.5477423384698731, 0.43624358954179776],
    dtype=np.float64,
)


def load_calibration():
    """Load base_link <- camera_link, falling back only if YAML is absent."""
    if not CALIBRATION_YAML.is_file():
        return (
            FALLBACK_CALIBRATION_TRANSLATION.copy(),
            FALLBACK_CALIBRATION_QUATERNION.copy(),
            "embedded fallback",
        )
    with CALIBRATION_YAML.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    parameters = document.get("parameters", {})
    if (
        parameters.get("eye_on_hand") is not False
        or parameters.get("robot_base_frame") != "base_link"
        or parameters.get("tracking_base_frame") != "camera_link"
    ):
        raise RuntimeError("saved YAML has the wrong calibration frames")
    transform = document["transformation"]
    translation = np.array(
        [transform[key] for key in ("x", "y", "z")], dtype=np.float64
    )
    quaternion = np.array(
        [transform[key] for key in ("qx", "qy", "qz", "qw")],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)):
        raise RuntimeError("saved YAML contains a non-finite transform")
    return translation, quaternion, str(CALIBRATION_YAML)


CALIBRATION_TRANSLATION, CALIBRATION_QUATERNION, CALIBRATION_SOURCE = (
    load_calibration()
)


def transform_message_to_matrix(message):
    """Convert geometry_msgs/TransformStamped to parent_T_child."""
    transform = message.transform
    quaternion = [
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w,
    ]
    matrix = quaternion_matrix(quaternion)
    matrix[:3, 3] = [
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    ]
    return matrix


def calibration_matrix():
    quaternion = CALIBRATION_QUATERNION.copy()
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("invalid calibration quaternion")
    matrix = quaternion_matrix(quaternion / norm)
    matrix[:3, 3] = CALIBRATION_TRANSLATION
    return matrix


def pose_values(matrix):
    xyz = matrix[:3, 3]
    quaternion = quaternion_from_matrix(matrix)  # x, y, z, w
    rpy = np.asarray(euler_from_matrix(matrix), dtype=np.float64)
    return xyz, quaternion, rpy, np.degrees(rpy)


def print_pose(title, matrix):
    xyz, quaternion, rpy, rpy_degrees = pose_values(matrix)
    print(title)
    print("  xyz [m]       : [{:.6f}, {:.6f}, {:.6f}]".format(*xyz))
    print("  quaternion xyzw: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(
        *quaternion
    ))
    print("  rpy [rad]     : [{:.6f}, {:.6f}, {:.6f}]".format(*rpy))
    print("  rpy [deg]     : [{:.3f}, {:.3f}, {:.3f}]".format(*rpy_degrees))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform live ArUco pose into the UR3 base frame"
    )
    parser.add_argument(
        "--once", action="store_true", help="print one measurement and exit"
    )
    parser.add_argument(
        "--rate", type=float, default=2.0, help="continuous output rate in Hz"
    )
    return parser.parse_known_args(rospy.myargv(argv=sys.argv)[1:])[0]


def main():
    args = parse_args()
    if not math.isfinite(args.rate) or args.rate <= 0.0:
        print("--rate must be a positive finite number", file=sys.stderr)
        return 2

    rospy.init_node("marker_to_base", anonymous=True)
    rospy.loginfo(
        "Hand-eye source=%s translation=%s quaternion_xyzw=%s",
        CALIBRATION_SOURCE,
        CALIBRATION_TRANSLATION.tolist(),
        CALIBRATION_QUATERNION.tolist(),
    )
    buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    listener = tf2_ros.TransformListener(buffer)

    base_t_camera_link = calibration_matrix()
    rate = rospy.Rate(args.rate)
    while not rospy.is_shutdown():
        try:
            # RealSense's fixed internal extrinsic.
            camera_link_t_optical = transform_message_to_matrix(
                buffer.lookup_transform(
                    "camera_link", "camera_color_optical_frame",
                    rospy.Time(0), rospy.Duration(3.0)
                )
            )
            # Live ArUco measurement.
            optical_t_marker = transform_message_to_matrix(
                buffer.lookup_transform(
                    "camera_color_optical_frame", "aruco_marker_frame",
                    rospy.Time(0), rospy.Duration(3.0)
                )
            )
            # Live UR3 end-effector pose.
            base_t_tool0 = transform_message_to_matrix(
                buffer.lookup_transform(
                    "base_link", "tool0", rospy.Time(0), rospy.Duration(3.0)
                )
            )

            camera_link_t_marker = camera_link_t_optical.dot(optical_t_marker)
            base_t_marker = base_t_camera_link.dot(camera_link_t_marker)

            print("\n" + "=" * 72)
            print_pose(
                "Marker in camera optical frame "
                "(camera_color_optical_frame <- aruco_marker_frame):",
                optical_t_marker,
            )
            print_pose(
                "Marker in robot base frame (base_link <- aruco_marker_frame):",
                base_t_marker,
            )
            print_pose(
                "Current robot end pose (base_link <- tool0):",
                base_t_tool0,
            )
            print("\nbase_link <- aruco_marker_frame matrix:")
            print(np.array2string(base_t_marker, precision=6, suppress_small=True))
            sys.stdout.flush()

            if args.once:
                return 0
            rate.sleep()
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(2.0, "Waiting for required TF: %s", exc)
            if args.once:
                return 1
            rate.sleep()

    return 0


if __name__ == "__main__":
    sys.exit(main())
