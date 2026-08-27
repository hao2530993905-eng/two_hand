#!/usr/bin/env python3
"""Validate saved D435 eye-on-hand calibration with the fixed ArUco 582."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import rospy
import tf2_ros
import yaml
from tf.transformations import euler_from_matrix, quaternion_from_matrix, quaternion_matrix


CALIBRATION_YAML = (
    Path.home() / ".ros/easy_handeye/d435_ur3_eye_on_hand.yaml"
)


def load_calibration(path):
    if not path.is_file():
        raise RuntimeError("calibration file does not exist: {}".format(path))
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    parameters = document.get("parameters", {})
    expected = {
        "eye_on_hand": True,
        "robot_base_frame": "base_link",
        "robot_effector_frame": "tool0",
        "tracking_base_frame": "d435_link",
        "tracking_marker_frame": "d435_aruco_marker_frame",
    }
    mismatches = [
        "{}={!r} (expected {!r})".format(key, parameters.get(key), value)
        for key, value in expected.items() if parameters.get(key) != value
    ]
    if mismatches:
        raise RuntimeError("wrong calibration YAML: " + "; ".join(mismatches))

    transform = document.get("transformation", {})
    try:
        translation = np.array(
            [float(transform[key]) for key in ("x", "y", "z")], dtype=float
        )
        quaternion = np.array(
            [float(transform[key]) for key in ("qx", "qy", "qz", "qw")],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("invalid transformation: {}".format(error))
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(quaternion)):
        raise RuntimeError("calibration contains non-finite values")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise RuntimeError("calibration quaternion has zero length")
    matrix = quaternion_matrix(quaternion / norm)
    matrix[:3, 3] = translation
    return matrix


def stamped_transform_to_matrix(message):
    transform = message.transform
    quaternion = np.array([
        transform.rotation.x, transform.rotation.y,
        transform.rotation.z, transform.rotation.w,
    ], dtype=float)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise RuntimeError("TF contains a zero-length quaternion")
    matrix = quaternion_matrix(quaternion / norm)
    matrix[:3, 3] = [
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    ]
    return matrix


def rotation_distance_deg(first, second):
    relative = first[:3, :3].T.dot(second[:3, :3])
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def print_pose(title, matrix):
    xyz = matrix[:3, 3]
    quaternion = quaternion_from_matrix(matrix)
    rpy_deg = np.degrees(np.asarray(euler_from_matrix(matrix), dtype=float))
    print(title)
    print("  xyz [m]        : [{:.6f}, {:.6f}, {:.6f}]".format(*xyz))
    print("  quaternion xyzw: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(
        *quaternion))
    print("  rpy [deg]      : [{:.3f}, {:.3f}, {:.3f}]".format(*rpy_deg))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check whether fixed marker 582 remains fixed in base_link")
    parser.add_argument("--once", action="store_true",
                        help="print one measurement and exit")
    parser.add_argument("--rate", type=float, default=2.0,
                        help="continuous output rate in Hz (default: %(default)s)")
    parser.add_argument("--calibration", type=Path, default=CALIBRATION_YAML,
                        help="D435 Easy Handeye YAML")
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


def main():
    args = parse_args()
    if not math.isfinite(args.rate) or args.rate <= 0.0:
        raise RuntimeError("--rate must be positive and finite")
    tool_t_d435 = load_calibration(args.calibration)

    rospy.init_node("d435_marker_to_base_validator", anonymous=True)
    rospy.loginfo("Using D435 calibration: %s", args.calibration)
    buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    listener = tf2_ros.TransformListener(buffer)
    rate = rospy.Rate(args.rate)
    reference = None

    while not rospy.is_shutdown():
        try:
            base_t_tool = stamped_transform_to_matrix(buffer.lookup_transform(
                "base_link", "tool0", rospy.Time(0), rospy.Duration(3.0)))
            d435_t_optical = stamped_transform_to_matrix(buffer.lookup_transform(
                "d435_link", "d435_color_optical_frame",
                rospy.Time(0), rospy.Duration(3.0)))
            optical_t_marker = stamped_transform_to_matrix(buffer.lookup_transform(
                "d435_color_optical_frame", "d435_aruco_marker_frame",
                rospy.Time(0), rospy.Duration(3.0)))
            base_t_marker = (
                base_t_tool.dot(tool_t_d435).dot(d435_t_optical).dot(optical_t_marker)
            )
            if reference is None:
                reference = base_t_marker.copy()

            translation_drift_mm = (
                np.linalg.norm(base_t_marker[:3, 3] - reference[:3, 3]) * 1000.0
            )
            rotation_drift_deg = rotation_distance_deg(reference, base_t_marker)
            print("\n" + "=" * 72)
            print_pose("582 in base_link:", base_t_marker)
            print("  drift from first: translation={:.3f} mm, rotation={:.3f} deg".format(
                translation_drift_mm, rotation_drift_deg))
            if args.once:
                return 0
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as error:
            rospy.logwarn_throttle(3.0, "Waiting for D435/UR3/marker TF: %s", error)
        rate.sleep()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, yaml.YAMLError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        sys.exit(1)
