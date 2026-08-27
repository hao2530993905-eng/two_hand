#!/usr/bin/env python3
"""Republish factory CameraInfo with the calibration loaded as ROS params."""

import copy
import math
import sys

import rospy
from sensor_msgs.msg import CameraInfo


def numeric_list(name, length):
    values = [float(value) for value in rospy.get_param(name)]
    if len(values) != length or not all(math.isfinite(value) for value in values):
        raise ValueError("{} must contain {} finite values".format(name, length))
    return values


class CameraInfoOverride:
    def __init__(self):
        self.width = int(rospy.get_param("~image_width"))
        self.height = int(rospy.get_param("~image_height"))
        self.model = str(rospy.get_param("~distortion_model"))
        self.d = [float(value) for value in rospy.get_param("~distortion_coefficients/data")]
        self.k = numeric_list("~camera_matrix/data", 9)
        self.r = numeric_list("~rectification_matrix/data", 9)
        self.p = numeric_list("~projection_matrix/data", 12)
        self.input_topic = str(rospy.get_param("~input_topic"))
        self.output_topic = str(rospy.get_param("~output_topic"))
        if self.width <= 0 or self.height <= 0 or self.input_topic == self.output_topic:
            raise ValueError("invalid image size or CameraInfo topics")
        if not self.d or not all(math.isfinite(value) for value in self.d):
            raise ValueError("invalid distortion coefficients")
        self.publisher = rospy.Publisher(self.output_topic, CameraInfo, queue_size=10)
        self.subscriber = rospy.Subscriber(
            self.input_topic, CameraInfo, self.callback, queue_size=10
        )
        self.logged = False

    def callback(self, source):
        if source.width != self.width or source.height != self.height:
            rospy.logerr_throttle(
                5.0, "CameraInfo size mismatch: stream=%dx%d calibration=%dx%d",
                source.width, source.height, self.width, self.height,
            )
            return
        result = copy.deepcopy(source)
        result.distortion_model = self.model
        result.D, result.K, result.R, result.P = self.d, self.k, self.r, self.p
        self.publisher.publish(result)
        if not self.logged:
            rospy.loginfo("CameraInfo calibration active on %s", self.output_topic)
            self.logged = True


def main():
    rospy.init_node("camera_info_override")
    try:
        CameraInfoOverride()
    except (KeyError, TypeError, ValueError, rospy.ROSException) as error:
        rospy.logfatal("Invalid camera calibration: %s", error)
        return 1
    rospy.spin()
    return 0


if __name__ == "__main__":
    sys.exit(main())
