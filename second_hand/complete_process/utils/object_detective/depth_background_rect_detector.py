#!/usr/bin/env python3
"""Detect raised white rectangular parts using aligned D435 depth images.

Remove the parts and press B to collect a multi-frame depth background. The
detector keeps pixels that are closer to the camera than that background by a
configurable height, intersects them with a white HSV mask, and fits rotated
rectangles. Accepted rectangles can be saved directly as YOLO-OBB samples.
"""

import json
import sys
import threading
import time
import warnings
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complete_process.utils.object_detective.rectangle_geometry import (  # noqa: E402
    detect_rectangles,
)


CONTROL_WINDOW = "Depth controls"
DEBUG_WINDOW = "Depth detection"
MASK_WINDOW = "Height mask"

def depth_to_millimeters(depth_message, bridge):
    depth = bridge.imgmsg_to_cv2(depth_message, desired_encoding="passthrough")
    if depth.dtype == np.uint16:
        return depth.copy()
    if depth.dtype in (np.float32, np.float64):
        result = np.zeros(depth.shape, dtype=np.uint16)
        valid = np.isfinite(depth) & (depth > 0.0)
        result[valid] = np.clip(depth[valid] * 1000.0, 0, 65535).astype(np.uint16)
        return result
    raise ValueError("unsupported depth encoding/dtype: {}/{}".format(
        depth_message.encoding, depth.dtype
    ))


def median_depth_background(frames):
    stack = np.stack(frames).astype(np.float32)
    stack[stack == 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        background = np.nanmedian(stack, axis=0)
    return np.nan_to_num(background, nan=0.0).astype(np.uint16)


def height_mask(current_depth, background_depth, min_height_mm,
                max_height_mm, median_size, open_size, close_size):
    if median_size >= 3:
        current_depth = cv2.medianBlur(current_depth, median_size | 1)

    valid = (current_depth > 0) & (background_depth > 0)
    height = background_depth.astype(np.int32) - current_depth.astype(np.int32)
    foreground = valid & (height >= min_height_mm)
    if max_height_mm > 0:
        foreground &= height <= max_height_mm
    mask = foreground.astype(np.uint8) * 255

    if open_size >= 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (open_size, open_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if close_size >= 2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                           (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, height


def white_color_mask(bgr_image, max_saturation, min_value):
    """Return pixels that are bright and weakly saturated in HSV space."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        (0, 0, int(min_value)),
        (179, int(max_saturation), 255),
    )


class DepthBackgroundRectangleDetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.color_topic = rospy.get_param(
            "~color_topic", "/d435/color/image_raw"
        )
        self.depth_topic = rospy.get_param(
            "~depth_topic", "/d435/aligned_depth_to_color/image_raw"
        )
        default_path = Path(__file__).resolve().parent / "depth_background.npy"
        self.background_path = Path(
            rospy.get_param("~background_path", str(default_path))
        ).expanduser()

        self.min_height_mm = int(rospy.get_param("~min_height_mm", 12))
        # The target is inserted into an open fixture with a distant wall
        # behind it, so its depth delta can be far greater than a tabletop
        # workpiece height. Zero intentionally disables the upper bound.
        self.max_height_mm = int(rospy.get_param("~max_height_mm", 0))
        self.background_samples = int(rospy.get_param("~background_samples", 20))
        self.median_size = int(rospy.get_param("~median_size", 5))
        self.open_size = int(rospy.get_param("~open_size", 3))
        self.close_size = int(rospy.get_param("~close_size", 11))
        self.min_area = float(rospy.get_param("~min_area", 1200))
        self.white_max_saturation = int(
            rospy.get_param("~white_max_saturation", 120)
        )
        self.white_min_value = int(
            rospy.get_param("~white_min_value", 90)
        )
        self.max_area = float(rospy.get_param("~max_area", 0))
        self.min_aspect = float(rospy.get_param("~min_aspect", 1.0))
        self.max_aspect = float(rospy.get_param("~max_aspect", 6.0))
        self.min_rectangularity = float(
            rospy.get_param("~min_rectangularity", 0.55)
        )
        self.min_solidity = float(rospy.get_param("~min_solidity", 0.60))
        if self.min_area < 0.0:
            raise ValueError("min_area must be non-negative")
        if self.max_area < 0.0:
            raise ValueError("max_area must be non-negative")
        if self.max_area and self.max_area < self.min_area:
            raise ValueError("max_area must be zero or greater than min_area")
        if not 0 <= self.white_max_saturation <= 255:
            raise ValueError("white_max_saturation must be in [0, 255]")
        if not 0 <= self.white_min_value <= 255:
            raise ValueError("white_min_value must be in [0, 255]")
        if not 1.0 <= self.min_aspect <= self.max_aspect:
            raise ValueError("aspect range must satisfy 1 <= min_aspect <= max_aspect")
        if not 0.0 <= self.min_rectangularity <= 1.0:
            raise ValueError("min_rectangularity must be in [0, 1]")
        if not 0.0 <= self.min_solidity <= 1.0:
            raise ValueError("min_solidity must be in [0, 1]")
        self.max_objects = int(rospy.get_param("~max_objects", 10))
        self.roi_x = int(rospy.get_param("~roi_x", 0))
        self.roi_y = int(rospy.get_param("~roi_y", 0))
        self.roi_width = int(rospy.get_param("~roi_width", 0))
        self.roi_height = int(rospy.get_param("~roi_height", 0))
        self.show_gui = bool(rospy.get_param("~show_gui", True))
        self.collect_dataset = bool(rospy.get_param("~collect_dataset", False))
        self.dataset_name = str(
            rospy.get_param("~dataset_name", "white_rectangle_dataset")
        ).strip()
        if (
            not self.dataset_name
            or Path(self.dataset_name).name != self.dataset_name
            or self.dataset_name in (".", "..")
        ):
            raise ValueError(
                "dataset_name must be a non-empty folder name without slashes"
            )
        default_dataset = Path(__file__).resolve().parent / self.dataset_name
        self.dataset_dir = Path(rospy.get_param(
            "~dataset_dir", str(default_dataset))).expanduser()
        self.dataset_split = str(rospy.get_param("~dataset_split", "train"))
        self.class_id = int(rospy.get_param("~class_id", 0))
        if self.dataset_split not in ("train", "val"):
            raise ValueError("dataset_split must be train or val")
        if self.class_id != 0:
            raise ValueError("this one-class dataset requires class_id=0")
        self.saved_samples = []

        self.background = None
        self.capture_frames = []
        self.capture_remaining = 0
        self.latest_pair = None
        self.pair_lock = threading.Lock()
        self.controls_initialized = False
        self.image_width = 0
        self.image_height = 0

        if self.background_path.is_file():
            try:
                self.background = np.load(str(self.background_path))
                if self.background.dtype != np.uint16 or self.background.ndim != 2:
                    raise ValueError("expected a 2-D uint16 depth array")
                rospy.loginfo("Loaded depth background: %s", self.background_path)
            except (OSError, ValueError) as error:
                self.background = None
                rospy.logwarn("Cannot load depth background: %s", error)

        self.corners_pub = rospy.Publisher("~corners", Float32MultiArray,
                                           queue_size=1)
        self.result_pub = rospy.Publisher("~result", String, queue_size=1)
        self.mask_pub = rospy.Publisher("~mask", Image, queue_size=1)
        self.debug_pub = rospy.Publisher("~debug_image", Image, queue_size=1)

        if self.show_gui:
            cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
            cv2.createTrackbar("Min height mm", CONTROL_WINDOW,
                               self.min_height_mm, 200, lambda _: None)
            cv2.createTrackbar("Max height mm", CONTROL_WINDOW,
                               self.max_height_mm, 500, lambda _: None)
            cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
            cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)

        color_sub = message_filters.Subscriber(self.color_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.10
        )
        self.synchronizer.registerCallback(self.pair_callback)
        rospy.on_shutdown(cv2.destroyAllWindows)
        rospy.loginfo("Color topic: %s", self.color_topic)
        rospy.loginfo("Aligned depth topic: %s", self.depth_topic)
        if self.collect_dataset:
            self.initialize_dataset()
            rospy.loginfo("Dataset collection enabled: %s (%s)",
                          self.dataset_dir, self.dataset_split)
        if self.background is None:
            rospy.logwarn("Remove parts, focus the detection window, then press B")

    def pair_callback(self, color_message, depth_message):
        with self.pair_lock:
            self.latest_pair = (color_message, depth_message)

    def initialize_controls(self, color):
        if not self.show_gui or self.controls_initialized:
            return
        self.image_height, self.image_width = color.shape[:2]
        x = max(0, min(self.roi_x, self.image_width - 1))
        y = max(0, min(self.roi_y, self.image_height - 1))
        width = self.image_width - x if self.roi_width <= 0 else min(
            self.roi_width, self.image_width - x
        )
        height = self.image_height - y if self.roi_height <= 0 else min(
            self.roi_height, self.image_height - y
        )
        cv2.createTrackbar("ROI x", CONTROL_WINDOW, x,
                           self.image_width - 1, lambda _: None)
        cv2.createTrackbar("ROI y", CONTROL_WINDOW, y,
                           self.image_height - 1, lambda _: None)
        cv2.createTrackbar("ROI width", CONTROL_WINDOW, width,
                           self.image_width, lambda _: None)
        cv2.createTrackbar("ROI height", CONTROL_WINDOW, height,
                           self.image_height, lambda _: None)
        self.controls_initialized = True

    def read_controls(self):
        self.min_height_mm = max(
            1, cv2.getTrackbarPos("Min height mm", CONTROL_WINDOW)
        )
        max_height = cv2.getTrackbarPos("Max height mm", CONTROL_WINDOW)
        self.max_height_mm = (
            0 if max_height == 0 else max(self.min_height_mm, max_height)
        )
        self.roi_x = cv2.getTrackbarPos("ROI x", CONTROL_WINDOW)
        self.roi_y = cv2.getTrackbarPos("ROI y", CONTROL_WINDOW)
        self.roi_width = max(
            1, cv2.getTrackbarPos("ROI width", CONTROL_WINDOW)
        )
        self.roi_height = max(
            1, cv2.getTrackbarPos("ROI height", CONTROL_WINDOW)
        )

    def roi_bounds(self, image):
        height, width = image.shape[:2]
        x1 = max(0, min(self.roi_x, width - 1))
        y1 = max(0, min(self.roi_y, height - 1))
        x2 = min(width, x1 + self.roi_width) if self.roi_width > 0 else width
        y2 = min(height, y1 + self.roi_height) if self.roi_height > 0 else height
        return x1, y1, x2, y2

    def start_background_capture(self):
        self.capture_frames = []
        self.capture_remaining = self.background_samples
        rospy.loginfo("Capturing %d empty-workspace depth frames...",
                      self.background_samples)

    def add_background_sample(self, depth):
        self.capture_frames.append(depth.copy())
        self.capture_remaining -= 1
        if self.capture_remaining > 0:
            return
        self.background = median_depth_background(self.capture_frames)
        self.background_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.background_path), self.background)
        self.capture_frames = []
        rospy.loginfo("Saved median depth background: %s", self.background_path)

    def reset_roi(self):
        cv2.setTrackbarPos("ROI x", CONTROL_WINDOW, 0)
        cv2.setTrackbarPos("ROI y", CONTROL_WINDOW, 0)
        cv2.setTrackbarPos("ROI width", CONTROL_WINDOW, self.image_width)
        cv2.setTrackbarPos("ROI height", CONTROL_WINDOW, self.image_height)

    def initialize_dataset(self):
        for category in ("images", "labels", "previews"):
            (self.dataset_dir / category / self.dataset_split).mkdir(
                parents=True, exist_ok=True)
        yaml_path = self.dataset_dir / "dataset.yaml"
        if not yaml_path.exists():
            yaml_path.write_text(
                "path: {}\ntrain: images/train\nval: images/val\n\n"
                "names:\n  0: white_rectangle\n".format(
                    self.dataset_dir.resolve()
                ),
                encoding="utf-8",
            )

    def sample_paths(self):
        name = "frame_{}_{}".format(int(time.time() * 1000), time.time_ns() % 1000000)
        return (
            self.dataset_dir / "images" / self.dataset_split / (name + ".jpg"),
            self.dataset_dir / "labels" / self.dataset_split / (name + ".txt"),
            self.dataset_dir / "previews" / self.dataset_split / (name + ".jpg"),
        )

    def save_dataset_sample(self, color, objects, negative=False):
        if not self.collect_dataset:
            rospy.logwarn("Restart with _collect_dataset:=true to save samples")
            return
        if not negative and not objects:
            rospy.logwarn("Not saved: no depth rectangle is currently detected")
            return

        image_path, label_path, preview_path = self.sample_paths()
        height, width = color.shape[:2]
        labels = []
        preview = color.copy()
        if negative:
            cv2.putText(preview, "NEGATIVE (empty label)", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            for obj in objects:
                corners = np.asarray(obj["corners_px"], dtype=np.float32)
                normalized = corners.copy()
                normalized[:, 0] = np.clip(normalized[:, 0] / width, 0.0, 1.0)
                normalized[:, 1] = np.clip(normalized[:, 1] / height, 0.0, 1.0)
                labels.append("{} {}".format(
                    self.class_id,
                    " ".join("{:.6f}".format(value)
                             for value in normalized.reshape(-1)),
                ))
                cv2.polylines(preview, [np.rint(corners).astype(np.int32)],
                              True, (0, 255, 0), 3)

        if not cv2.imwrite(str(image_path), color,
                           [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError("cannot save image: {}".format(image_path))
        label_path.write_text("\n".join(labels) + ("\n" if labels else ""),
                              encoding="utf-8")
        if not cv2.imwrite(str(preview_path), preview,
                           [cv2.IMWRITE_JPEG_QUALITY, 95]):
            image_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
            raise OSError("cannot save preview: {}".format(preview_path))
        self.saved_samples.append((image_path, label_path, preview_path))
        rospy.loginfo("Saved %s sample: %s (%d labels)",
                      "negative" if negative else "positive",
                      image_path.name, len(labels))

    def undo_last_sample(self):
        if not self.saved_samples:
            rospy.logwarn("Nothing saved in this session to undo")
            return
        paths = self.saved_samples.pop()
        for path in paths:
            path.unlink(missing_ok=True)
        rospy.loginfo("Removed last sample: %s", paths[0].name)

    def process_latest(self):
        with self.pair_lock:
            pair = self.latest_pair
            self.latest_pair = None
        if pair is None:
            rospy.logwarn_throttle(
                3.0, "Waiting for synchronized color and aligned depth images ..."
            )
            if self.show_gui:
                cv2.waitKey(1)
            return

        color_message, depth_message = pair
        try:
            color = self.bridge.imgmsg_to_cv2(color_message, "bgr8")
            depth = depth_to_millimeters(depth_message, self.bridge)
            if color.shape[:2] != depth.shape:
                raise ValueError("color/depth dimensions differ: {} vs {}".format(
                    color.shape[:2], depth.shape
                ))
            self.initialize_controls(color)
            if self.show_gui:
                self.read_controls()

            if self.capture_remaining > 0:
                self.add_background_sample(depth)

            debug = color.copy()
            x1, y1, x2, y2 = self.roi_bounds(color)
            cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 0), 2)
            keys = "B: background  R: reset ROI"
            if self.collect_dataset:
                keys += "  S: positive  N: negative  D: undo"
            cv2.putText(debug, keys,
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 0, 0), 2)
            height_text = (
                "{}..{} mm".format(self.min_height_mm, self.max_height_mm)
                if self.max_height_mm > 0
                else ">={} mm (no max)".format(self.min_height_mm)
            )
            cv2.putText(
                debug,
                "ROI {} {} {} {}  height={}  white=S<={} V>={}  min_area={:.0f}".format(
                    x1, y1, x2 - x1, y2 - y1,
                    height_text,
                    self.white_max_saturation, self.white_min_value,
                    self.min_area),
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 0, 0), 2,
            )

            full_mask = np.zeros(depth.shape, dtype=np.uint8)
            detections = []
            if self.capture_remaining > 0:
                cv2.putText(debug, "CAPTURING BACKGROUND: {} frames left".format(
                    self.capture_remaining), (10, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 128, 255), 2)
            elif self.background is None:
                cv2.putText(debug, "REMOVE PARTS AND PRESS B", (10, 88),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            elif self.background.shape != depth.shape:
                rospy.logerr_throttle(3.0, "Depth background size differs; press B")
            else:
                mask, _ = height_mask(
                    depth[y1:y2, x1:x2],
                    self.background[y1:y2, x1:x2],
                    self.min_height_mm, self.max_height_mm,
                    self.median_size, self.open_size, self.close_size,
                )
                color_mask = white_color_mask(
                    color[y1:y2, x1:x2],
                    self.white_max_saturation,
                    self.white_min_value,
                )
                mask = cv2.bitwise_and(mask, color_mask)
                full_mask[y1:y2, x1:x2] = mask
                detections = detect_rectangles(
                    mask,
                    self.min_area,
                    self.max_objects,
                    max_area=self.max_area,
                    min_aspect=self.min_aspect,
                    max_aspect=self.max_aspect,
                    min_rectangularity=self.min_rectangularity,
                    min_solidity=self.min_solidity,
                )

            objects = []
            flat_corners = []
            offset = np.array((x1, y1), dtype=np.float32)
            for index, detection in enumerate(detections):
                corners = detection["corners"] + offset
                center = detection["center"] + offset
                integer_corners = np.rint(corners).astype(np.int32)
                cv2.polylines(debug, [integer_corners], True, (0, 255, 0), 3)
                cv2.putText(debug, "#{} A={:.2f} R={:.2f}".format(
                    index, detection["aspect"], detection["rectangularity"]),
                    tuple(np.rint(center).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                flat_corners.extend(corners.reshape(-1).tolist())
                objects.append({
                    "id": len(objects),
                    "area_px": round(detection["area"], 2),
                    "aspect_ratio": round(detection["aspect"], 4),
                    "rectangularity": round(detection["rectangularity"], 4),
                    "solidity": round(detection.get("solidity", 0.0), 4),
                    "center_px": center.round(2).tolist(),
                    "angle_deg": round(detection["angle"], 3),
                    "corners_px": corners.round(2).tolist(),
                })

            result = {
                "background_ready": self.background is not None,
                "count": len(objects),
                "height_range_mm": [self.min_height_mm, self.max_height_mm],
                "target_color": "white",
                "white_hsv": {
                    "max_saturation": self.white_max_saturation,
                    "min_value": self.white_min_value,
                },
                "min_area_px": self.min_area,
                "geometry_filter": {
                    "max_area_px": self.max_area,
                    "min_aspect": self.min_aspect,
                    "max_aspect": self.max_aspect,
                    "min_rectangularity": self.min_rectangularity,
                    "min_solidity": self.min_solidity,
                },
                "roi": {"x": x1, "y": y1, "width": x2 - x1,
                        "height": y2 - y1},
                "objects": objects,
            }
            self.corners_pub.publish(Float32MultiArray(data=flat_corners))
            self.result_pub.publish(String(data=json.dumps(result)))
            mask_message = self.bridge.cv2_to_imgmsg(full_mask, "mono8")
            mask_message.header = depth_message.header
            self.mask_pub.publish(mask_message)
            debug_message = self.bridge.cv2_to_imgmsg(debug, "bgr8")
            debug_message.header = color_message.header
            self.debug_pub.publish(debug_message)
            rospy.loginfo_throttle(1.0, "Detected %d depth rectangle(s)",
                                   len(objects))

            if self.show_gui:
                cv2.imshow(MASK_WINDOW, full_mask)
                cv2.imshow(DEBUG_WINDOW, debug)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("b"), ord("B")):
                    self.start_background_capture()
                elif key in (ord("r"), ord("R")):
                    self.reset_roi()
                elif key in (ord("s"), ord("S")):
                    self.save_dataset_sample(color, objects, negative=False)
                elif key in (ord("n"), ord("N")):
                    self.save_dataset_sample(color, objects, negative=True)
                elif key in (ord("d"), ord("D")):
                    self.undo_last_sample()
        except (CvBridgeError, cv2.error, OSError, ValueError) as error:
            rospy.logerr_throttle(2.0, "Depth detector error: %s", error)


def main():
    rospy.init_node("depth_background_rect_detector")
    detector = DepthBackgroundRectangleDetector()
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        detector.process_latest()
        rate.sleep()


if __name__ == "__main__":
    main()
