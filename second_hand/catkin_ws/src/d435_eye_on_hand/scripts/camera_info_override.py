#!/usr/bin/env python3
"""Republish factory CameraInfo with the measured D435 color intrinsics."""

import copy
import math
import sys

import rospy
from sensor_msgs.msg import CameraInfo


def numeric_list(parameter, expected_length):
    values = [float(value) for value in rospy.get_param(parameter)]
    if len(values) != expected_length or not all(map(math.isfinite, values)):
        raise ValueError("{} must contain {} finite values".format(parameter, expected_length))
    return values


class CameraInfoOverride:
    def __init__(self):
        self.width = int(rospy.get_param("~image_width"))
        self.height = int(rospy.get_param("~image_height"))
        self.distortion_model = rospy.get_param("~distortion_model")
        self.k = numeric_list("~camera_matrix/data", 9)
        self.d = [float(value) for value in rospy.get_param("~distortion_coefficients/data")]
        self.r = numeric_list("~rectification_matrix/data", 9)
        self.p = numeric_list("~projection_matrix/data", 12)
        if not self.d or not all(map(math.isfinite, self.d)):
            raise ValueError("distortion coefficients must be finite and non-empty")
        self.input_topic = rospy.get_param("~input_topic")
        self.output_topic = rospy.get_param("~output_topic")
        if self.width <= 0 or self.height <= 0 or self.input_topic == self.output_topic:
            raise ValueError("invalid dimensions or topics")
        self.publisher = rospy.Publisher(self.output_topic, CameraInfo, queue_size=10)
        self.subscriber = rospy.Subscriber(self.input_topic, CameraInfo, self.callback, queue_size=10)
        self.logged = False

    def callback(self, source):
        if source.width != self.width or source.height != self.height:
            rospy.logerr_throttle(
                5.0,
                "CameraInfo size mismatch: stream=%dx%d calibration=%dx%d",
                source.width, source.height, self.width, self.height,
            )
            return
        result = copy.deepcopy(source)
        result.distortion_model = self.distortion_model
        result.D, result.K, result.R, result.P = self.d, self.k, self.r, self.p
        self.publisher.publish(result)
        if not self.logged:
            rospy.loginfo("Using measured D435 %dx%d color intrinsics", self.width, self.height)
            self.logged = True


def main():
    rospy.init_node("d435_color_camera_info_override")
    try:
        CameraInfoOverride()
    except (KeyError, TypeError, ValueError, rospy.ROSException) as exc:
        rospy.logfatal("Invalid D435 camera calibration: %s", exc)
        return 1
    rospy.spin()
    return 0


if __name__ == "__main__":
    sys.exit(main())

