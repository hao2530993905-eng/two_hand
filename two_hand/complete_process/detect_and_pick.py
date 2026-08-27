#!/usr/bin/env python3
"""Detect one random workpiece with YOLO-OBB and move 3 cm above it.

The node performs RGB inference itself, obtains metric Z from aligned depth,
deprojects the OBB center with the color-camera intrinsics, and transforms the
point from the camera optical frame to base_link through the calibrated TF.

The saved Easy Handeye YAML is loaded as the default calibration, so a
separate Easy Handeye TF publisher is not required for normal operation.

For safety this is a dry run unless --execute is supplied.  Even with
--execute, the operator must type MOVE unless --yes is also supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

# Load Conda binary modules before exposing Debian ROS Python packages.  This
# avoids cv2/numpy ABI conflicts when the YOLO Conda environment is used.
import cv2
import numpy as np
import yaml
from ultralytics import YOLO

if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")

import rospy
import tf2_ros
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf.transformations import quaternion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "complete_process/utils/object_detective/models/red_box.pt"
)

# Fallback used only when the Easy Handeye YAML does not exist.  Transform
# direction: base_link <- camera_link; quaternion order: ROS [x, y, z, w].
FALLBACK_HAND_EYE_TRANSLATION = (
    -0.01044787244394433,
    0.4078309180113589,
    0.8614295017631832,
)
FALLBACK_HAND_EYE_QUATERNION = (
    -0.49690966616336485,
    0.512592084317204,
    0.5477423384698731,
    0.43624358954179776,
)
DEFAULT_HAND_EYE_YAML = (
    PROJECT_ROOT / "config/ur3_d455_handeye_eye_on_base.yaml"
)


def load_saved_handeye_calibration(
    path: Path = DEFAULT_HAND_EYE_YAML,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float], str]:
    """Load and validate base_link <- camera_link from Easy Handeye YAML."""
    if not path.is_file():
        return (
            FALLBACK_HAND_EYE_TRANSLATION,
            FALLBACK_HAND_EYE_QUATERNION,
            "embedded fallback ({} not found)".format(path),
        )

    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise RuntimeError("invalid Easy Handeye YAML: {}".format(path))
    parameters = document.get("parameters", {})
    if (
        parameters.get("eye_on_hand") is not False
        or parameters.get("robot_base_frame") != "base_link"
        or parameters.get("tracking_base_frame") != "camera_link"
    ):
        raise RuntimeError(
            "{} is not the expected eye-on-base base_link <- camera_link "
            "calibration".format(path)
        )
    transform = document.get("transformation", {})
    try:
        translation = tuple(float(transform[key]) for key in ("x", "y", "z"))
        quaternion = tuple(
            float(transform[key]) for key in ("qx", "qy", "qz", "qw")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "{} has an invalid transformation: {}".format(path, error)
        )
    values = np.asarray(translation + quaternion, dtype=np.float64)
    quaternion_norm = float(np.linalg.norm(quaternion))
    if not np.all(np.isfinite(values)) or quaternion_norm < 1e-12:
        raise RuntimeError("{} contains a non-finite transform".format(path))
    return translation, quaternion, str(path)


(
    DEFAULT_HAND_EYE_TRANSLATION,
    DEFAULT_HAND_EYE_QUATERNION,
    DEFAULT_HAND_EYE_SOURCE,
) = load_saved_handeye_calibration()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complete_process.utils.base.ur_base import UR_BASE  # noqa: E402


def stamp_seconds(message: Image) -> float:
    return float(message.header.stamp.to_sec())


def color_message_to_bgr(message: Image) -> np.ndarray:
    if message.encoding not in ("bgr8", "rgb8"):
        raise ValueError(
            "color encoding must be bgr8 or rgb8, got {}".format(message.encoding)
        )
    row_bytes = message.width * 3
    raw = np.frombuffer(message.data, dtype=np.uint8)
    image = raw.reshape(message.height, message.step)[:, :row_bytes]
    image = image.reshape(message.height, message.width, 3).copy()
    if message.encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def depth_message_to_meters(message: Image) -> np.ndarray:
    encoding = message.encoding.upper()
    if encoding in ("16UC1", "MONO16"):
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
        scale = 0.001
    elif encoding == "32FC1":
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
        scale = 1.0
    else:
        raise ValueError("unsupported depth encoding: {}".format(message.encoding))
    row_values = message.step // dtype.itemsize
    raw = np.frombuffer(message.data, dtype=dtype)
    depth = raw.reshape(message.height, row_values)[:, :message.width]
    return depth.astype(np.float64) * scale


def bgr_to_image_message(image: np.ndarray, header: Any) -> Image:
    message = Image()
    message.header = header
    message.height, message.width = image.shape[:2]
    message.encoding = "bgr8"
    message.is_bigendian = 0
    message.step = message.width * 3
    message.data = image.tobytes()
    return message


def transform_to_matrix(transform: Any) -> np.ndarray:
    rotation = transform.rotation
    matrix = quaternion_matrix(
        [rotation.x, rotation.y, rotation.z, rotation.w]
    )
    translation = transform.translation
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


def calibration_to_matrix(
    translation: Sequence[float], quaternion: Sequence[float]
) -> np.ndarray:
    """Build base <- camera matrix from XYZ and ROS XYZW quaternion."""
    xyz = np.asarray(translation, dtype=np.float64)
    xyzw = np.asarray(quaternion, dtype=np.float64)
    if xyz.shape != (3,) or xyzw.shape != (4,):
        raise ValueError("hand-eye translation/quaternion has invalid shape")
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(xyzw)):
        raise ValueError("hand-eye calibration contains non-finite values")
    norm = float(np.linalg.norm(xyzw))
    if norm < 1e-12:
        raise ValueError("hand-eye quaternion has zero length")
    xyzw /= norm
    matrix = quaternion_matrix(xyzw)
    matrix[:3, 3] = xyz
    return matrix


def robust_depth(
    depth_m: np.ndarray,
    u: float,
    v: float,
    radius: int,
    min_depth: float,
    max_depth: float,
) -> Tuple[float, int]:
    """Return median valid depth near the target center."""
    height, width = depth_m.shape
    column, row = int(round(u)), int(round(v))
    if not 0 <= column < width or not 0 <= row < height:
        raise ValueError("YOLO center is outside the aligned depth image")
    x1, x2 = max(0, column - radius), min(width, column + radius + 1)
    y1, y2 = max(0, row - radius), min(height, row + radius + 1)
    patch = depth_m[y1:y2, x1:x2]
    valid = patch[
        np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)
    ]
    if valid.size == 0:
        raise ValueError("no valid aligned depth near the YOLO center")
    return float(np.median(valid)), int(valid.size)


def deproject_pixel(
    u: float, v: float, depth: float, camera_info: CameraInfo
) -> np.ndarray:
    fx, fy = float(camera_info.K[0]), float(camera_info.K[4])
    cx, cy = float(camera_info.K[2]), float(camera_info.K[5])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("invalid color-camera intrinsics")
    return np.array(
        [(u - cx) * depth / fx, (v - cy) * depth / fy, depth, 1.0],
        dtype=np.float64,
    )


def parse_bounds(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(values, dtype=np.float64)
    low, high = bounds[[0, 2, 4]], bounds[[1, 3, 5]]
    if not np.all(np.isfinite(bounds)) or np.any(low >= high):
        raise ValueError("invalid workspace bounds")
    return low, high


def base_link_point_to_ur_base(point: Sequence[float]) -> np.ndarray:
    """Convert REP-103 base_link XYZ to the UR controller Base coordinates.

    universal_robot defines base_link -> base as a fixed pi rotation about Z,
    so X and Y change sign while Z is unchanged. RTDE TCP poses use `base`.
    """
    xyz = np.asarray(point, dtype=np.float64)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("base_link point must contain three finite values")
    return np.array([-xyz[0], -xyz[1], xyz[2]], dtype=np.float64)


ROBOT_MODE_NAMES = {
    -1: "NO_CONTROLLER",
    0: "DISCONNECTED",
    1: "CONFIRM_SAFETY",
    2: "BOOTING",
    3: "POWER_OFF",
    4: "POWER_ON",
    5: "IDLE",
    6: "BACKDRIVE",
    7: "RUNNING",
    8: "UPDATING_FIRMWARE",
}

SAFETY_MODE_NAMES = {
    1: "NORMAL",
    2: "REDUCED",
    3: "PROTECTIVE_STOP",
    4: "RECOVERY",
    5: "SAFEGUARD_STOP",
    6: "SYSTEM_EMERGENCY_STOP",
    7: "ROBOT_EMERGENCY_STOP",
    8: "VIOLATION",
    9: "FAULT",
    10: "VALIDATE_JOINT_ID",
    11: "UNDEFINED_SAFETY_MODE",
    12: "AUTOMATIC_MODE_SAFEGUARD_STOP",
    13: "SYSTEM_THREE_POSITION_ENABLING_STOP",
}


def verify_robot_ready(robot: UR_BASE) -> List[float]:
    """Fail before motion unless the controller is in a safe running state."""
    if robot.rtde_r is None:
        raise RuntimeError("RTDE receive interface is unavailable")
    robot_mode = int(robot.rtde_r.getRobotMode())
    safety_mode = int(robot.rtde_r.getSafetyMode())
    robot_name = ROBOT_MODE_NAMES.get(robot_mode, "UNKNOWN")
    safety_name = SAFETY_MODE_NAMES.get(safety_mode, "UNKNOWN")
    print(
        "robot state: mode={}({}), safety={}({})".format(
            robot_mode, robot_name, safety_mode, safety_name
        )
    )
    if safety_mode != 1:
        raise RuntimeError(
            "robot safety mode is {}({}), not NORMAL(1); clear the safety "
            "condition on the teach pendant before retrying".format(
                safety_mode, safety_name
            )
        )
    if robot_mode != 7:
        raise RuntimeError(
            "robot mode is {}({}), not RUNNING(7); on the teach pendant "
            "power on the robot and release the brakes before retrying".format(
                robot_mode, robot_name
            )
        )
    return robot.get_tcp_pose_or_raise()


class SensorCollector:
    """Keep the newest color frame and a short aligned-depth history."""

    def __init__(
        self, color_topic: str, depth_topic: str, camera_info_topic: str
    ) -> None:
        self.condition = threading.Condition()
        self.color_message: Optional[Image] = None
        self.color_sequence = 0
        self.depth_messages: Deque[Image] = deque(maxlen=60)
        self.camera_info: Optional[CameraInfo] = None
        self.color_sub = rospy.Subscriber(
            color_topic, Image, self.color_callback, queue_size=1, buff_size=2**24
        )
        self.depth_sub = rospy.Subscriber(
            depth_topic, Image, self.depth_callback, queue_size=5, buff_size=2**24
        )
        self.info_sub = rospy.Subscriber(
            camera_info_topic, CameraInfo, self.info_callback, queue_size=1
        )

    def color_callback(self, message: Image) -> None:
        with self.condition:
            self.color_message = message
            self.color_sequence += 1
            self.condition.notify_all()

    def depth_callback(self, message: Image) -> None:
        with self.condition:
            self.depth_messages.append(message)
            self.condition.notify_all()

    def info_callback(self, message: CameraInfo) -> None:
        with self.condition:
            self.camera_info = message
            self.condition.notify_all()

    def next_bundle(
        self, previous_sequence: int, deadline: float, max_sync_delta: float
    ) -> Tuple[int, Image, Image, CameraInfo, float]:
        """Wait for a new color frame and return its nearest depth frame."""
        with self.condition:
            while not rospy.is_shutdown():
                if (
                    self.color_sequence != previous_sequence
                    and self.color_message is not None
                    and self.depth_messages
                    and self.camera_info is not None
                ):
                    color = self.color_message
                    color_stamp = stamp_seconds(color)
                    depth = min(
                        self.depth_messages,
                        key=lambda item: abs(stamp_seconds(item) - color_stamp),
                    )
                    delta = abs(stamp_seconds(depth) - color_stamp)
                    if delta <= max_sync_delta:
                        return (
                            self.color_sequence,
                            color,
                            depth,
                            self.camera_info,
                            delta,
                        )
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "timed out waiting for synchronized color, aligned depth, "
                        "and camera_info"
                    )
                self.condition.wait(min(remaining, 0.1))
        raise RuntimeError("ROS stopped while waiting for camera data")

    def next_color(
        self, previous_sequence: int, timeout: float = 1.0
    ) -> Tuple[int, Image]:
        """Wait for a new color frame without requiring depth synchronization."""
        deadline = time.monotonic() + timeout
        with self.condition:
            while not rospy.is_shutdown():
                if (
                    self.color_sequence != previous_sequence
                    and self.color_message is not None
                ):
                    return self.color_sequence, self.color_message
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for a color image")
                self.condition.wait(min(remaining, 0.1))
        raise RuntimeError("ROS stopped while waiting for a color image")


class YoloTargetAcquirer:
    def __init__(self, args: argparse.Namespace, collector: SensorCollector) -> None:
        self.args = args
        self.collector = collector
        self.model = YOLO(str(args.model))
        self.debug_pub = rospy.Publisher(
            "~debug_image", Image, queue_size=1, latch=True
        )
        self.result_pub = rospy.Publisher(
            "~result", String, queue_size=1, latch=True
        )
        self.preview_stop = threading.Event()
        self.preview_thread: Optional[threading.Thread] = None
        if args.show_gui:
            rospy.logwarn(
                "--show-gui is ignored to avoid the Conda OpenCV/ROS Qt "
                "conflict; view /detect_and_pick/debug_image with "
                "rqt_image_view instead"
            )

    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        prediction = self.model.predict(
            image,
            imgsz=self.args.imgsz,
            conf=self.args.confidence,
            iou=self.args.iou,
            device=self.args.device,
            verbose=False,
        )[0]
        objects: List[Dict[str, Any]] = []
        if prediction.obb is None:
            return objects
        corners = prediction.obb.xyxyxyxy.cpu().numpy()
        boxes = prediction.obb.xywhr.cpu().numpy()
        scores = prediction.obb.conf.cpu().numpy()
        for index, (points, box, score) in enumerate(zip(corners, boxes, scores)):
            center_x, center_y, width, height, angle = map(float, box)
            objects.append(
                {
                    "id": index,
                    "confidence": float(score),
                    "center_px": [center_x, center_y],
                    "size_px": [width, height],
                    "angle_deg": math.degrees(angle),
                    "corners_px": points.astype(np.float64),
                }
            )
        return objects

    def draw_and_publish(
        self,
        image: np.ndarray,
        header: Any,
        objects: List[Dict[str, Any]],
        selected: Optional[Dict[str, Any]],
    ) -> None:
        debug = image.copy()
        for item in objects:
            is_selected = item is selected
            color = (0, 0, 255) if is_selected else (0, 255, 0)
            thickness = 4 if is_selected else 2
            points = np.rint(item["corners_px"]).astype(np.int32)
            cv2.polylines(debug, [points], True, color, thickness)
            u, v = item["center_px"]
            cv2.circle(debug, (int(round(u)), int(round(v))), 5, color, -1)
            label = "{}#{:d} {:.2f}".format(
                "SELECTED " if is_selected else "",
                int(item["id"]),
                item["confidence"],
            )
            cv2.putText(
                debug,
                label,
                (int(points[0, 0]), int(points[0, 1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        cv2.putText(
            debug,
            "objects={}  red=random target".format(len(objects)),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )
        self.debug_pub.publish(bgr_to_image_message(debug, header))

    def acquire(self) -> Dict[str, Any]:
        deadline = time.monotonic() + self.args.input_timeout
        previous_sequence = -1
        tracked_center: Optional[np.ndarray] = None
        samples: List[Dict[str, Any]] = []

        while not rospy.is_shutdown() and len(samples) < self.args.stable_samples:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out after {:.1f}s without acquiring {} stable "
                    "YOLO samples; inspect /detect_and_pick/debug_image and "
                    "check that the workpiece is visible to the camera".format(
                        self.args.input_timeout, self.args.stable_samples
                    )
                )
            sequence, color_msg, depth_msg, info, sync_delta = (
                self.collector.next_bundle(
                    previous_sequence, deadline, self.args.max_sync_delta
                )
            )
            previous_sequence = sequence
            image = color_message_to_bgr(color_msg)
            objects = self.infer(image)
            selected: Optional[Dict[str, Any]] = None

            if objects:
                if tracked_center is None:
                    selected = random.choice(objects)
                    rospy.loginfo(
                        "Randomly selected target %d of %d at [%.2f, %.2f]",
                        int(selected["id"]) + 1,
                        len(objects),
                        selected["center_px"][0],
                        selected["center_px"][1],
                    )
                else:
                    selected = min(
                        objects,
                        key=lambda item: np.linalg.norm(
                            np.asarray(item["center_px"]) - tracked_center
                        ),
                    )
                    association_distance = float(
                        np.linalg.norm(
                            np.asarray(selected["center_px"]) - tracked_center
                        )
                    )
                    if association_distance > self.args.association_px:
                        selected = None

            if selected is None:
                tracked_center = None
                samples.clear()
                self.draw_and_publish(image, color_msg.header, objects, None)
                if not objects:
                    rospy.logwarn_throttle(
                        2.0,
                        "No YOLO workpiece detected (confidence >= %.2f); "
                        "inspect /detect_and_pick/debug_image",
                        self.args.confidence,
                    )
                else:
                    rospy.logwarn_throttle(
                        2.0,
                        "Detected objects, but the tracked target moved more "
                        "than %.1f px; restarting stable sampling",
                        self.args.association_px,
                    )
                continue

            center = np.asarray(selected["center_px"], dtype=np.float64)
            if (
                tracked_center is not None
                and np.linalg.norm(center - tracked_center) > self.args.stability_px
            ):
                samples.clear()
            tracked_center = center

            try:
                depth_m = depth_message_to_meters(depth_msg)
                depth, valid_count = robust_depth(
                    depth_m,
                    center[0],
                    center[1],
                    self.args.depth_radius,
                    self.args.min_depth,
                    self.args.max_depth,
                )
                camera_point = deproject_pixel(center[0], center[1], depth, info)
            except ValueError as error:
                rospy.logwarn_throttle(1.0, "Rejected target depth: %s", error)
                samples.clear()
                self.draw_and_publish(image, color_msg.header, objects, selected)
                continue

            sample = dict(selected)
            sample["camera_point_m"] = camera_point[:3]
            sample["depth_m"] = depth
            sample["depth_valid_count"] = valid_count
            sample["sync_delta_s"] = sync_delta
            samples.append(sample)
            self.draw_and_publish(image, color_msg.header, objects, selected)
            rospy.loginfo(
                "Stable target sample %d/%d: center=[%.2f, %.2f], depth=%.4f m",
                len(samples),
                self.args.stable_samples,
                center[0],
                center[1],
                depth,
            )

        if not samples:
            raise RuntimeError("ROS stopped before a stable YOLO target was acquired")

        result = dict(samples[-1])
        result["center_px"] = np.mean(
            [sample["center_px"] for sample in samples], axis=0
        ).tolist()
        result["camera_point_m"] = np.mean(
            [sample["camera_point_m"] for sample in samples], axis=0
        ).tolist()
        result["depth_m"] = float(
            np.median([sample["depth_m"] for sample in samples])
        )
        result["confidence"] = float(
            np.mean([sample["confidence"] for sample in samples])
        )
        result["depth_valid_count"] = int(
            np.mean([sample["depth_valid_count"] for sample in samples])
        )
        # Keep the accepted frames for downstream robust geometry estimation.
        # Consumers that do not need multi-frame refinement can ignore it.
        result["_stable_samples"] = samples
        return result

    def start_live_preview(self, locked_target: Dict[str, Any]) -> None:
        """Continue publishing YOLO frames while main waits or moves the robot."""
        if self.preview_thread is not None and self.preview_thread.is_alive():
            return
        self.preview_stop.clear()
        locked_center = np.asarray(locked_target["center_px"], dtype=np.float64)

        def preview_loop() -> None:
            previous_sequence = -1
            rospy.loginfo(
                "Live preview active: rqt_image_view /detect_and_pick/debug_image"
            )
            while not rospy.is_shutdown() and not self.preview_stop.is_set():
                try:
                    sequence, color_msg = self.collector.next_color(
                        previous_sequence, timeout=1.0
                    )
                    previous_sequence = sequence
                    image = color_message_to_bgr(color_msg)
                    objects = self.infer(image)
                    selected = None
                    if objects:
                        nearest = min(
                            objects,
                            key=lambda item: np.linalg.norm(
                                np.asarray(item["center_px"]) - locked_center
                            ),
                        )
                        if (
                            np.linalg.norm(
                                np.asarray(nearest["center_px"]) - locked_center
                            )
                            <= self.args.association_px
                        ):
                            selected = nearest
                    self.draw_and_publish(
                        image, color_msg.header, objects, selected
                    )
                except TimeoutError:
                    rospy.logwarn_throttle(
                        3.0,
                        "Live preview is waiting for %s",
                        self.args.color_topic,
                    )
                except (RuntimeError, ValueError) as error:
                    rospy.logwarn_throttle(2.0, "Live preview error: %s", error)

        self.preview_thread = threading.Thread(
            target=preview_loop, name="yolo_live_preview", daemon=True
        )
        self.preview_thread.start()

    def stop_live_preview(self) -> None:
        self.preview_stop.set()
        if self.preview_thread is not None:
            self.preview_thread.join(timeout=2.0)

    def publish_final_result(
        self,
        target: Dict[str, Any],
        base_point: np.ndarray,
        base_link_target: np.ndarray,
        target_pose: List[float],
    ) -> None:
        payload = {
            "confidence": round(float(target["confidence"]), 4),
            "center_px": np.asarray(target["center_px"]).round(3).tolist(),
            "camera_point_m": np.asarray(target["camera_point_m"]).round(6).tolist(),
            "base_surface_point_m": base_point.round(6).tolist(),
            "base_link_target_xyz_m": base_link_target.round(6).tolist(),
            "ur_base_target_tcp_pose": [round(value, 6) for value in target_pose],
            "height_above_object_m": self.args.above_height,
        }
        self.result_pub.publish(String(data=json.dumps(payload)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO-OBB, randomly select one workpiece, and move the UR3 "
            "to 3 cm above its detected surface center."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--color-topic", default="/camera/color/image_raw")
    parser.add_argument(
        "--depth-topic", default="/camera/aligned_depth_to_color/image_raw"
    )
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument("--handeye-camera-frame", default="camera_link")
    parser.add_argument(
        "--calibration-source",
        choices=("embedded", "tf"),
        default="embedded",
        help=(
            "use the saved/default base<-camera_link hand-eye result plus "
            "the RealSense internal TF (default), or look up "
            "--base-frame <- --camera-frame entirely from TF"
        ),
    )
    parser.add_argument(
        "--handeye-translation",
        nargs=3,
        type=float,
        default=DEFAULT_HAND_EYE_TRANSLATION,
        metavar=("X", "Y", "Z"),
        help="saved/default base<-handeye-camera-frame translation in metres",
    )
    parser.add_argument(
        "--handeye-quaternion",
        nargs=4,
        type=float,
        default=DEFAULT_HAND_EYE_QUATERNION,
        metavar=("QX", "QY", "QZ", "QW"),
        help="saved/default base<-handeye-camera-frame quaternion in ROS x y z w order",
    )
    parser.add_argument("--host", default="192.168.1.4")
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--stable-samples", type=int, default=8)
    parser.add_argument("--stability-px", type=float, default=5.0)
    parser.add_argument("--association-px", type=float, default=80.0)
    parser.add_argument("--input-timeout", type=float, default=30.0)
    parser.add_argument(
        "--tf-timeout",
        type=float,
        default=10.0,
        help="seconds to wait for the calibrated base-to-camera TF",
    )
    parser.add_argument("--max-sync-delta", type=float, default=0.10)
    parser.add_argument("--depth-radius", type=int, default=5)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument(
        "--above-height",
        type=float,
        default=0.03,
        help="base-Z distance above the detected workpiece surface in metres",
    )
    parser.add_argument(
        "--target-offset",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help="optional base-frame XYZ correction applied before above-height",
    )
    parser.add_argument(
        "--tcp-rotation",
        nargs=3,
        type=float,
        default=(-0.028553, 3.090451, -0.127509),
        metavar=("RX", "RY", "RZ"),
        help="fixed downward-facing TCP rotation vector in UR controller base",
    )
    parser.add_argument(
        "--workspace",
        nargs=6,
        type=float,
        default=(-0.45, 0.45, 0.10, 0.55, 0.02, 0.55),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="allowed target TCP bounds in base_link, metres",
    )
    parser.add_argument("--joint-speed", type=float, default=0.20)
    parser.add_argument("--joint-acc", type=float, default=0.35)
    parser.add_argument("--motion-timeout", type=float, default=25.0)
    parser.add_argument(
        "--max-start-distance",
        type=float,
        default=0.0,
        help=(
            "optional start-to-target translation limit in metres; "
            "0 disables this check"
        ),
    )
    parser.add_argument("--random-seed", type=int)
    parser.add_argument(
        "--show-gui",
        action="store_true",
        help=(
            "compatibility option; GUI is intentionally disabled, use "
            "rqt_image_view on /detect_and_pick/debug_image"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually connect and move; otherwise only calculate and print",
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the interactive MOVE confirmation"
    )
    parser.add_argument(
        "--no-live-preview",
        action="store_true",
        help="disable continuous /detect_and_pick/debug_image publishing",
    )
    args = parser.parse_args(rospy.myargv()[1:])

    if not args.model.is_file():
        parser.error("model does not exist: {}".format(args.model))
    if not 0.0 < args.confidence <= 1.0 or not 0.0 < args.iou <= 1.0:
        parser.error("--confidence and --iou must be in (0, 1]")
    if args.imgsz <= 0 or args.stable_samples <= 0 or args.depth_radius < 0:
        parser.error("--imgsz and --stable-samples must be positive; depth radius >= 0")
    if args.stability_px <= 0.0 or args.association_px <= args.stability_px:
        parser.error("--association-px must be greater than positive --stability-px")
    if not 0.0 < args.min_depth < args.max_depth:
        parser.error("require 0 < --min-depth < --max-depth")
    for name in (
        "input_timeout",
        "tf_timeout",
        "max_sync_delta",
        "above_height",
        "joint_speed",
        "joint_acc",
        "motion_timeout",
    ):
        if getattr(args, name) <= 0.0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.max_start_distance < 0.0:
        parser.error("--max-start-distance must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    if args.random_seed is not None:
        random.seed(args.random_seed)
    rospy.init_node("detect_and_pick")
    collector = SensorCollector(
        args.color_topic, args.depth_topic, args.camera_info_topic
    )
    acquirer = YoloTargetAcquirer(args, collector)
    rospy.loginfo("YOLO model: %s", args.model)
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    tf_listener = tf2_ros.TransformListener(tf_buffer)  # noqa: F841
    if args.calibration_source == "embedded":
        base_to_handeye_camera = calibration_to_matrix(
            args.handeye_translation, args.handeye_quaternion
        )
        uses_default = np.allclose(
            np.asarray(args.handeye_translation, dtype=np.float64),
            np.asarray(DEFAULT_HAND_EYE_TRANSLATION, dtype=np.float64),
        ) and np.allclose(
            np.asarray(args.handeye_quaternion, dtype=np.float64),
            np.asarray(DEFAULT_HAND_EYE_QUATERNION, dtype=np.float64),
        )
        rospy.loginfo(
            "Using hand-eye calibration from %s: %s <- %s; "
            "waiting for %s <- %s...",
            DEFAULT_HAND_EYE_SOURCE if uses_default else "command-line values",
            args.base_frame,
            args.handeye_camera_frame,
            args.handeye_camera_frame,
            args.camera_frame,
        )
        try:
            camera_internal = tf_buffer.lookup_transform(
                args.handeye_camera_frame,
                args.camera_frame,
                rospy.Time(0),
                rospy.Duration(args.tf_timeout),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            raise RuntimeError(
                "RealSense internal TF {} <- {} is unavailable: {}. Start "
                "the realsense2_camera node with TF enabled.".format(
                    args.handeye_camera_frame, args.camera_frame, error
                )
            )
        base_to_camera = base_to_handeye_camera @ transform_to_matrix(
            camera_internal.transform
        )
    else:
        rospy.loginfo(
            "Waiting for calibrated TF %s <- %s...",
            args.base_frame,
            args.camera_frame,
        )
        try:
            transform = tf_buffer.lookup_transform(
                args.base_frame,
                args.camera_frame,
                rospy.Time(0),
                rospy.Duration(args.tf_timeout),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as error:
            raise RuntimeError(
                "calibrated TF {} <- {} is unavailable: {}. Start the "
                "easy_handeye result publisher on the same ROS_MASTER_URI, "
                "or use --calibration-source embedded".format(
                    args.base_frame, args.camera_frame, error
                )
            )
        base_to_camera = transform_to_matrix(transform.transform)

    print("\n手眼标定外参（{} <- {}）：".format(
        args.base_frame, args.camera_frame
    ))
    print("  来源: {}".format(args.calibration_source))
    print("  变换矩阵:\n{}".format(np.array2string(base_to_camera, precision=8)))
    rospy.loginfo("Waiting for a stable randomly selected workpiece...")
    target = acquirer.acquire()
    camera_point = np.append(
        np.asarray(target["camera_point_m"], dtype=np.float64), 1.0
    )

    base_point = (base_to_camera @ camera_point)[:3]
    base_link_target_xyz = base_point + np.asarray(
        args.target_offset, dtype=np.float64
    )
    base_link_target_xyz[2] += args.above_height
    ur_base_target_xyz = base_link_point_to_ur_base(base_link_target_xyz)
    target_pose = ur_base_target_xyz.tolist() + [
        float(x) for x in args.tcp_rotation
    ]

    u, v = target["center_px"]
    camera_xyz = np.asarray(target["camera_point_m"], dtype=np.float64)
    print("\n" + "=" * 72)
    print("选中的 YOLO 目标")
    print("  置信度:                         {:.4f}".format(target["confidence"]))
    print("  图像中心 [u, v] px:             [{:.2f}, {:.2f}]".format(u, v))
    print("  深度 Z:                          {:.6f} m".format(target["depth_m"]))
    print(
        "  相机坐标系目标点 [Xc,Yc,Zc] m:  {}".format(
            camera_xyz.round(6).tolist()
        )
    )
    print(
        "  机械臂基坐标目标点 [Xb,Yb,Zb] m: {}".format(
            base_point.round(6).tolist()
        )
    )
    print("  目标上方高度:                    {:.3f} m".format(args.above_height))
    print("  base_link 上方 3cm 目标 XYZ [m]: {}".format(
        base_link_target_xyz.round(6).tolist()
    ))
    print("  UR 控制器 base 目标 TCP [x,y,z,rx,ry,rz]: {}".format(
        [round(x, 6) for x in target_pose]
    ))
    print("=" * 72)

    low, high = parse_bounds(args.workspace)
    if np.any(base_link_target_xyz < low) or np.any(base_link_target_xyz > high):
        raise ValueError(
            "base_link target point {} is outside workspace [{}, {}]".format(
                base_link_target_xyz.round(6).tolist(), low.tolist(), high.tolist()
            )
        )

    acquirer.publish_final_result(
        target, base_point, base_link_target_xyz, target_pose
    )
    if not args.no_live_preview:
        acquirer.start_live_preview(target)

    if not args.execute:
        print("DRY RUN：机械臂未连接、未运动。")
        if not args.no_live_preview:
            print("实时图像正在发布。另开终端运行：")
            print("  rqt_image_view /detect_and_pick/debug_image")
            print("按 Ctrl+C 结束预览。")
            try:
                rospy.spin()
            finally:
                acquirer.stop_live_preview()
        return
    if not args.yes:
        confirmation = input(
            "Type MOVE to move the robot to the printed target pose: "
        ).strip()
        if confirmation != "MOVE":
            print("Cancelled; robot was not connected or moved.")
            return

    with UR_BASE(args.host, gripper_port=None) as robot:
        current_pose = verify_robot_ready(robot)
        start_distance = float(
            np.linalg.norm(
                np.asarray(current_pose[:3], dtype=np.float64)
                - np.asarray(target_pose[:3], dtype=np.float64)
            )
        )
        print("current TCP pose: {}".format([round(x, 6) for x in current_pose]))
        print("start-to-target translation: {:.3f} m".format(start_distance))
        if (
            args.max_start_distance > 0.0
            and start_distance > args.max_start_distance
        ):
            raise RuntimeError(
                "start-to-target translation {:.3f} m exceeds the {:.3f} m "
                "safety limit; verify calibration and move the robot to a "
                "known observation pose first".format(
                    start_distance, args.max_start_distance
                )
            )
        print("\n发送运动命令前的目标坐标：")
        print("  相机坐标系目标点 [m]: {}".format(camera_xyz.round(6).tolist()))
        print("  机械臂基坐标目标点 [m]: {}".format(base_point.round(6).tolist()))
        print("  base_link 接近点 [m]:   {}".format(
            base_link_target_xyz.round(6).tolist()
        ))
        print("  UR base 目标 TCP 位姿:  {}".format(
            [round(x, 6) for x in target_pose]
        ))
        motion = robot.move_j_ik(
            target_pose,
            args.joint_speed,
            args.joint_acc,
            args.motion_timeout,
        )
        print("moveJ result: {}".format(motion))
        if not motion.success:
            raise RuntimeError("moveJ failed: {}".format(motion.message))
        actual_pose = robot.get_tcp_pose_or_raise()
        position_error = float(
            np.linalg.norm(
                np.asarray(actual_pose[:3]) - np.asarray(target_pose[:3])
            )
        )
        print("\n运动完成后的坐标：")
        print("  目标 TCP 位姿: {}".format([round(x, 6) for x in target_pose]))
        print("  实际 TCP 位姿: {}".format([round(x, 6) for x in actual_pose]))
        print("  位置误差:       {:.3f} mm".format(position_error * 1000.0))
    acquirer.stop_live_preview()
    print("Motion completed: TCP is 3 cm above the selected workpiece surface.")


if __name__ == "__main__":
    main()
