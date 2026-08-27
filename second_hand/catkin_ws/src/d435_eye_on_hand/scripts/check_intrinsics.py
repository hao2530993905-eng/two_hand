#!/usr/bin/env python3
"""Check that live CameraInfo exactly matches the installed D435 YAML."""

import math
import sys

import rospy
from sensor_msgs.msg import CameraInfo, Image


def close_list(left, right):
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-7)
        for a, b in zip(left, right)
    )


def main():
    rospy.init_node("check_d435_intrinsics", anonymous=True)
    try:
        image = rospy.wait_for_message("/d435/color/image_raw", Image, timeout=8.0)
        info = rospy.wait_for_message("/d435/color/camera_info", CameraInfo, timeout=8.0)
    except rospy.ROSException as exc:
        rospy.logerr("Missing D435 color stream: %s", exc)
        return 1
    expected_width = rospy.get_param("/d435_color_camera_info_override/image_width")
    expected_height = rospy.get_param("/d435_color_camera_info_override/image_height")
    expected_k = rospy.get_param("/d435_color_camera_info_override/camera_matrix/data")
    expected_d = rospy.get_param("/d435_color_camera_info_override/distortion_coefficients/data")
    ok = (
        image.width == info.width == int(expected_width)
        and image.height == info.height == int(expected_height)
        and info.header.frame_id == "d435_color_optical_frame"
        and close_list(info.K, expected_k)
        and close_list(info.D, expected_d)
    )
    if not ok:
        rospy.logerr("Live CameraInfo does not match installed D435 intrinsics")
        return 1
    rospy.loginfo("D435 measured intrinsics are active and resolution-consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
