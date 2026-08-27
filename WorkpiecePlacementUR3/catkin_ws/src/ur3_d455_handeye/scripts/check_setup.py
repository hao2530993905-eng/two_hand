#!/usr/bin/env python3
"""Check the four measurements required before taking a hand-eye sample."""

import math
import sys

import rospy
import tf2_ros
from sensor_msgs.msg import CameraInfo, Image, JointState


def wait_message(topic, message_type, timeout=5.0):
    try:
        message = rospy.wait_for_message(topic, message_type, timeout=timeout)
        rospy.loginfo("OK topic %s", topic)
        return message
    except rospy.ROSException as exc:
        rospy.logerr("Missing topic %s: %s", topic, exc)
        return None


def values_match(actual, expected, tolerance=1e-7):
    return len(actual) == len(expected) and all(
        math.isclose(float(a), float(e), rel_tol=tolerance, abs_tol=tolerance)
        for a, e in zip(actual, expected)
    )


def main():
    rospy.init_node("ur3_d455_handeye_check", anonymous=True)
    ok = True
    ok &= wait_message("/camera/color/image_raw", Image) is not None
    info = wait_message("/camera/color/camera_info", CameraInfo)
    ok &= info is not None
    if info is not None:
        if info.header.frame_id != "camera_color_optical_frame":
            rospy.logerr("Unexpected camera frame: %s", info.header.frame_id)
            ok = False
        else:
            rospy.loginfo("OK camera frame camera_color_optical_frame")
        expected_width = rospy.get_param(
            "/d455_color_camera_info_override/image_width", None
        )
        expected_height = rospy.get_param(
            "/d455_color_camera_info_override/image_height", None
        )
        expected_k = rospy.get_param(
            "/d455_color_camera_info_override/camera_matrix/data", None
        )
        expected_d = rospy.get_param(
            "/d455_color_camera_info_override/distortion_coefficients/data",
            None,
        )
        if None in (expected_width, expected_height, expected_k, expected_d):
            rospy.logerr("D455 CameraInfo override parameters are unavailable")
            ok = False
        elif (
            info.width != int(expected_width)
            or info.height != int(expected_height)
            or not values_match(info.K, expected_k)
            or not values_match(info.D, expected_d)
        ):
            rospy.logerr(
                "Live CameraInfo does not match the configured D455 intrinsics: "
                "size=%dx%d K=%s D=%s",
                info.width,
                info.height,
                list(info.K),
                list(info.D),
            )
            ok = False
        else:
            rospy.loginfo("OK D455 color intrinsics override")
    ok &= wait_message("/joint_states", JointState) is not None

    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)  # keep alive until checks finish
    rospy.sleep(1.0)
    for parent, child in (
        ("base_link", "tool0"),
        ("camera_link", "camera_color_frame"),
        ("camera_link", "camera_color_optical_frame"),
        ("camera_color_optical_frame", "aruco_marker_frame"),
    ):
        try:
            buffer.lookup_transform(parent, child, rospy.Time(0), rospy.Duration(5.0))
            rospy.loginfo("OK TF %s <- %s", parent, child)
        except Exception as exc:  # tf2 exception hierarchy differs among ROS releases
            rospy.logerr("Missing TF %s <- %s: %s", parent, child, exc)
            ok = False

    if ok:
        rospy.loginfo("Setup is ready for sampling.")
        return 0
    rospy.logerr("Setup is not ready; resolve the errors above before sampling.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
