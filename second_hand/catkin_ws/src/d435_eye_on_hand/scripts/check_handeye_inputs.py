#!/usr/bin/env python3
"""Check every live measurement required by eye-on-hand sampling."""

import sys

import rospy
import tf2_ros
from sensor_msgs.msg import CameraInfo, Image, JointState


def wait(topic, message_type):
    try:
        rospy.wait_for_message(topic, message_type, timeout=8.0)
        rospy.loginfo("OK topic %s", topic)
        return True
    except rospy.ROSException as exc:
        rospy.logerr("Missing topic %s: %s", topic, exc)
        return False


def main():
    rospy.init_node("check_eye_on_hand_inputs", anonymous=True)
    ok = wait("/d435/color/image_rect_color", Image)
    ok &= wait("/d435/color/camera_info", CameraInfo)
    ok &= wait("/joint_states", JointState)
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer)
    rospy.sleep(1.0)
    for parent, child in (
        ("base_link", "tool0"),
        ("d435_link", "d435_color_optical_frame"),
        ("d435_link", "d435_aruco_marker_frame"),
    ):
        try:
            buffer.lookup_transform(parent, child, rospy.Time(0), rospy.Duration(8.0))
            rospy.loginfo("OK TF %s <- %s", parent, child)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            rospy.logerr("Missing TF %s <- %s: %s", parent, child, exc)
            ok = False
    del listener
    if ok:
        rospy.loginfo("Eye-on-hand inputs are ready; robot motion remains manual")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
