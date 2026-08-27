#!/usr/bin/env python3
"""Run YOLO-OBB on a ROS image topic and publish an annotated image."""

import sys
from pathlib import Path

# The Conda interpreter does not include Ubuntu's rospkg/catkin_pkg packages.
# Appending (instead of prepending) keeps Conda's OpenCV ahead of Ubuntu's
# OpenCV, which avoids a libffi conflict while still allowing rospy to import.
UBUNTU_DIST_PACKAGES = "/usr/lib/python3/dist-packages"
if UBUNTU_DIST_PACKAGES not in sys.path:
    sys.path.append(UBUNTU_DIST_PACKAGES)

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "models" / "manual_white_rectangle_best.pt"


def ros_image_to_bgr(message):
    """Convert common 8-bit ROS image encodings without cv_bridge."""
    encoding = message.encoding.lower()
    channels_by_encoding = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
        "8uc1": 1,
        "8uc3": 3,
        "8uc4": 4,
    }
    if encoding not in channels_by_encoding:
        raise ValueError("unsupported image encoding: {}".format(message.encoding))

    channels = channels_by_encoding[encoding]
    packed_row_size = message.width * channels
    if message.step < packed_row_size:
        raise ValueError("image step is smaller than the packed row size")

    raw = np.frombuffer(message.data, dtype=np.uint8)
    required_size = message.height * message.step
    if raw.size < required_size:
        raise ValueError("image data is shorter than height * step")
    image = raw[:required_size].reshape(message.height, message.step)
    image = image[:, :packed_row_size]
    if channels == 1:
        image = image.reshape(message.height, message.width)
    else:
        image = image.reshape(message.height, message.width, channels)

    if encoding in ("rgb8",):
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding in ("rgba8",):
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding in ("bgra8", "8uc4"):
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding in ("mono8", "8uc1"):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(image)


def bgr_to_ros_image(image, source_header):
    """Create a bgr8 sensor_msgs/Image while preserving timestamp/frame ID."""
    contiguous = np.ascontiguousarray(image, dtype=np.uint8)
    message = Image()
    message.header = source_header
    message.height, message.width = contiguous.shape[:2]
    message.encoding = "bgr8"
    message.is_bigendian = 0
    message.step = message.width * 3
    message.data = contiguous.tobytes()
    return message


class YoloObbRosDetector:
    def __init__(self):
        model_path = Path(
            rospy.get_param("~model", str(DEFAULT_MODEL))
        ).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError("YOLO model does not exist: {}".format(model_path))

        self.image_topic = rospy.get_param(
            "~image_topic", "/d435/color/image_raw"
        )
        self.confidence = float(rospy.get_param("~conf", 0.25))
        self.iou = float(rospy.get_param("~iou", 0.45))
        self.image_size = int(rospy.get_param("~imgsz", 640))
        self.device = str(rospy.get_param("~device", "cpu"))
        self.max_detections = int(rospy.get_param("~max_det", 10))
        self.line_width = int(rospy.get_param("~line_width", 3))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("conf must be in [0, 1]")
        if not 0.0 <= self.iou <= 1.0:
            raise ValueError("iou must be in [0, 1]")
        if self.image_size <= 0 or self.max_detections <= 0:
            raise ValueError("imgsz and max_det must be positive")

        rospy.loginfo("Loading YOLO-OBB model: %s", model_path)
        self.model = YOLO(str(model_path))
        self.debug_publisher = rospy.Publisher(
            "~debug_image", Image, queue_size=1
        )
        self.subscriber = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )
        self.processed_frames = 0
        rospy.loginfo("Subscribing to: %s", self.image_topic)
        rospy.loginfo("Publishing annotated images to: %s", rospy.resolve_name("~debug_image"))
        rospy.loginfo(
            "Parameters: conf=%.2f iou=%.2f imgsz=%d device=%s",
            self.confidence,
            self.iou,
            self.image_size,
            self.device,
        )

    def image_callback(self, message):
        try:
            frame = ros_image_to_bgr(message)
            result = self.model.predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou,
                imgsz=self.image_size,
                device=self.device,
                max_det=self.max_detections,
                verbose=False,
            )[0]
            annotated = result.plot(line_width=self.line_width)
            self.debug_publisher.publish(
                bgr_to_ros_image(annotated, message.header)
            )
            self.processed_frames += 1
            if self.processed_frames == 1:
                rospy.loginfo("First annotated frame published")
        except (ValueError, RuntimeError, cv2.error) as error:
            rospy.logerr_throttle(2.0, "YOLO image processing failed: %s", error)


def main():
    rospy.init_node("white_rectangle_yolo_obb")
    try:
        YoloObbRosDetector()
        rospy.spin()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        rospy.logfatal("Cannot start YOLO-OBB detector: %s", error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
