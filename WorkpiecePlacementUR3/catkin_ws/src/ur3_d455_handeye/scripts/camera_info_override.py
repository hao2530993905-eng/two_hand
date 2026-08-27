#!/usr/bin/env python3
"""Publish D455 CameraInfo with project-calibrated color intrinsics."""

import copy
import math
import sys

import rospy
from sensor_msgs.msg import CameraInfo


def numeric_list(parameter, expected_length):
    values = [float(value) for value in rospy.get_param(parameter)]
    if len(values) != expected_length or not all(map(math.isfinite, values)):
        raise ValueError(
            "{} must contain {} finite values".format(
                parameter, expected_length
            )
        )
    return values


class CameraInfoOverride:
    def __init__(self):
        self.width = int(rospy.get_param("~image_width"))
        self.height = int(rospy.get_param("~image_height"))
        self.distortion_model = rospy.get_param("~distortion_model")
        self.k = numeric_list("~camera_matrix/data", 9)
        self.d = numeric_list("~distortion_coefficients/data", 5)
        self.r = numeric_list("~rectification_matrix/data", 9)
        self.p = numeric_list("~projection_matrix/data", 12)
        self.input_topic = rospy.get_param(
            "~input_topic", "/camera/color/camera_info_factory"
        )
        self.output_topic = rospy.get_param(
            "~output_topic", "/camera/color/camera_info"
        )
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if self.input_topic == self.output_topic:
            raise ValueError("input_topic and output_topic must differ")

        self.publisher = rospy.Publisher(
            self.output_topic, CameraInfo, queue_size=10
        )
        self.subscriber = rospy.Subscriber(
            self.input_topic, CameraInfo, self.callback, queue_size=10
        )
        self.logged = False

    def callback(self, source):
        if source.width != self.width or source.height != self.height:
            rospy.logerr_throttle(
                5.0,
                "Refusing CameraInfo override: source is %dx%d, calibration is %dx%d",
                source.width,
                source.height,
                self.width,
                self.height,
            )
            return

        result = copy.deepcopy(source)
        result.distortion_model = self.distortion_model
        result.D = self.d
        result.K = self.k
        result.R = self.r
        result.P = self.p
        self.publisher.publish(result)

        if not self.logged:
            rospy.loginfo(
                "Overriding %s -> %s with D455 %dx%d intrinsics: "
                "fx=%.5f fy=%.5f cx=%.5f cy=%.5f",
                self.input_topic,
                self.output_topic,
                self.width,
                self.height,
                self.k[0],
                self.k[4],
                self.k[2],
                self.k[5],
            )
            self.logged = True


def main():
    rospy.init_node("d455_color_camera_info_override")
    try:
        CameraInfoOverride()
    except (KeyError, TypeError, ValueError, rospy.ROSException) as exc:
        rospy.logfatal("Invalid D455 camera calibration: %s", exc)
        return 1
    rospy.spin()
    return 0


if __name__ == "__main__":
    sys.exit(main())
