#!/usr/bin/env python3
"""Detect one white rectangle and grasp it with an eye-on-hand D435.

Workflow:
1. Move to the calibrated observation TCP pose.
2. Capture and robustly fuse 12 stable YOLO-OBB RGB-D detections by default.
3. Deproject the OBB center and transform it through
   UR-base <- TCP <- d435_link <- color_optical_frame.
4. Align the undirected finger-to-finger line with the UR Base Z axis.
5. MoveJ to 10 cm in front of the target, MoveL to 5 cm in front, and close
   the Robotiq gripper with speed=255 and force=255.
6. MoveL upward along base +Z, then MoveJ/IK to the configured final TCP pose.

The program is a dry run unless --execute is supplied. In execution mode it
waits for two Enter confirmations unless --full-auto is also supplied.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

# Load Conda binary packages before Debian ROS packages to avoid cv2 ABI
# conflicts in the YOLO environment.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from complete_process.utils.base.gripper_orientation import (  # noqa: E402
    finger_line_in_tcp,
    matrix_to_rotvec,
    rotvec_to_matrix,
    rotation_distance,
)
from complete_process.utils.base.ur_base import UR_BASE  # noqa: E402


DEFAULT_MODEL = (
    PROJECT_ROOT
    / "complete_process/utils/object_detective/models/manual_white_rectangle_best.pt"
)
DEFAULT_HANDEYE = (
    PROJECT_ROOT / "config/d435_ur3_eye_on_hand.yaml"
)
# Nominal D435 d435_link <- color optical transform from
# realsense2_description/urdf/_d435.urdf.xacro. The camera driver normally
# publishes a device-calibrated equivalent, but some launch combinations omit
# /tf and /tf_static while still publishing color/aligned-depth images.
DEFAULT_D435_DEPTH_TO_COLOR_OFFSET_M = 0.015
OBSERVATION_POSE = np.array(
    [0.073120525, 0.463129264, 0.21510289, -1.284355, 1.309481, 1.488382],
    dtype=np.float64,
)
DEFAULT_FINAL_POSE = np.array(
    [-0.323390523, 0.120168609, 0.364210170, -1.489018, -1.057418, 1.446190],
    dtype=np.float64,
)


def normalize(vector: Sequence[float], name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("{} must contain three finite values".format(name))
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        raise ValueError("{} has near-zero length".format(name))
    return value / norm


def quaternion_to_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("quaternion has zero length")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )


def pose_to_matrix(pose: Sequence[float]) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("TCP pose must contain six finite values")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotvec_to_matrix(values[3:])
    matrix[:3, 3] = values[:3]
    return matrix


def transform_message_to_matrix(transform: Any) -> np.ndarray:
    matrix = quaternion_to_matrix(
        [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ]
    )
    matrix[:3, 3] = [
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
    ]
    return matrix


def nominal_d435_from_color_optical(color_offset_m: float) -> np.ndarray:
    """Return nominal d435_link <- d435_color_optical_frame."""
    offset = float(color_offset_m)
    if not math.isfinite(offset) or abs(offset) > 0.10:
        raise ValueError("D435 color offset must be finite and within 0.10 m")
    # URDF: d435_link -> color_frame has xyz=[0, 0.015, 0], rpy=0;
    # color_frame -> color_optical_frame has rpy=[-pi/2, 0, -pi/2].
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    matrix[:3, 3] = [0.0, offset, 0.0]
    return matrix


def load_tool_from_d435(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError("hand-eye calibration does not exist: {}".format(path))
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    parameters = document.get("parameters", {}) if isinstance(document, dict) else {}
    expected = {
        "eye_on_hand": True,
        "robot_effector_frame": "tool0",
        "tracking_base_frame": "d435_link",
    }
    mismatches = [
        "{}={!r}".format(key, parameters.get(key))
        for key, value in expected.items()
        if parameters.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "wrong eye-on-hand calibration metadata: {}".format(", ".join(mismatches))
        )
    transform = document.get("transformation", {})
    try:
        xyz = [float(transform[key]) for key in ("x", "y", "z")]
        quaternion = [float(transform[key]) for key in ("qx", "qy", "qz", "qw")]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid hand-eye transformation: {}".format(error))
    matrix = quaternion_to_matrix(quaternion)
    matrix[:3, 3] = xyz
    return matrix


def color_message_to_bgr(message: Image) -> np.ndarray:
    if message.encoding not in ("bgr8", "rgb8"):
        raise ValueError("unsupported color encoding: {}".format(message.encoding))
    packed = message.width * 3
    raw = np.frombuffer(message.data, dtype=np.uint8)
    required = message.height * message.step
    if raw.size < required or message.step < packed:
        raise ValueError("invalid color image step/data size")
    image = raw[:required].reshape(message.height, message.step)[:, :packed]
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
    if message.step % dtype.itemsize:
        raise ValueError("depth step is not aligned to its element size")
    row_values = message.step // dtype.itemsize
    raw = np.frombuffer(message.data, dtype=dtype)
    required = message.height * row_values
    if raw.size < required or row_values < message.width:
        raise ValueError("invalid depth image step/data size")
    depth = raw[:required].reshape(message.height, row_values)[:, : message.width]
    return depth.astype(np.float64) * scale


def bgr_to_image_message(image: np.ndarray, header: Any) -> Image:
    contiguous = np.ascontiguousarray(image, dtype=np.uint8)
    message = Image()
    message.header = header
    message.height, message.width = contiguous.shape[:2]
    message.encoding = "bgr8"
    message.is_bigendian = 0
    message.step = message.width * 3
    message.data = contiguous.tobytes()
    return message


def stamp_seconds(message: Image) -> float:
    return float(message.header.stamp.to_sec())


def robust_depth(
    depth_m: np.ndarray,
    u: float,
    v: float,
    radius: int,
    min_depth: float,
    max_depth: float,
) -> Tuple[float, int]:
    height, width = depth_m.shape
    column, row = int(round(u)), int(round(v))
    if not 0 <= column < width or not 0 <= row < height:
        raise ValueError("OBB center lies outside the aligned depth image")
    x1, x2 = max(0, column - radius), min(width, column + radius + 1)
    y1, y2 = max(0, row - radius), min(height, row + radius + 1)
    patch = depth_m[y1:y2, x1:x2]
    valid = patch[
        np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)
    ]
    if valid.size == 0:
        raise ValueError("no valid aligned depth near the OBB center")
    return float(np.median(valid)), int(valid.size)


def deproject_pixel(
    u: float, v: float, depth: float, camera_info: CameraInfo
) -> np.ndarray:
    fx, fy = float(camera_info.K[0]), float(camera_info.K[4])
    cx, cy = float(camera_info.K[2]), float(camera_info.K[5])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera_info contains invalid focal lengths")
    return np.array(
        [(u - cx) * depth / fx, (v - cy) * depth / fy, depth],
        dtype=np.float64,
    )


def long_edge_pixels(corners: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(corners, dtype=np.float64)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise ValueError("OBB corners must be a finite 4x2 array")
    next_points = np.roll(points, -1, axis=0)
    lengths = np.linalg.norm(next_points - points, axis=1)
    index = int(np.argmax(lengths))
    if lengths[index] < 2.0:
        raise ValueError("OBB long edge is too short")
    return points[index], next_points[index]


def tcp_from_gripper_rotation() -> np.ndarray:
    """Build the calibrated mounting rotation, assuming TCP Z = gripper Z."""
    line_tcp = finger_line_in_tcp().copy()
    line_tcp[2] = 0.0
    gripper_x = normalize(line_tcp, "projected finger line in TCP")
    gripper_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    gripper_y = normalize(np.cross(gripper_z, gripper_x), "gripper Y in TCP")
    return np.column_stack((gripper_x, gripper_y, gripper_z))


def desired_tcp_rotation(
    approach_axis_in_base: Sequence[float],
    nearby_tcp_rotation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align the undirected finger line exactly with the UR Base Z axis.

    Gripper X is the calibrated finger-to-finger line. Because gripper X and
    its approach axis must be orthogonal, the camera-to-target direction is
    projected onto the Base XY plane and used as gripper Z.
    """
    gripper_x = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    raw_approach = normalize(approach_axis_in_base, "camera-to-target axis")
    gripper_z = raw_approach - float(np.dot(raw_approach, gripper_x)) * gripper_x
    gripper_z = normalize(gripper_z, "approach axis projected onto Base XY")
    tcp_from_gripper = tcp_from_gripper_rotation()
    candidates = []
    for sign in (1.0, -1.0):
        x_axis = sign * gripper_x
        y_axis = normalize(np.cross(gripper_z, x_axis), "gripper Y in base")
        base_from_gripper = np.column_stack((x_axis, y_axis, gripper_z))
        base_from_tcp = base_from_gripper @ tcp_from_gripper.T
        candidates.append((base_from_tcp, x_axis))
    rotation, selected_axis = min(
        candidates,
        key=lambda item: rotation_distance(nearby_tcp_rotation, item[0]),
    )
    return rotation, selected_axis, gripper_z


class SensorCollector:
    def __init__(self, color_topic: str, depth_topic: str, info_topic: str) -> None:
        self.condition = threading.Condition()
        self.color: Optional[Image] = None
        self.color_sequence = 0
        self.depths: Deque[Image] = deque(maxlen=60)
        self.info: Optional[CameraInfo] = None
        self.color_sub = rospy.Subscriber(
            color_topic, Image, self._color_callback, queue_size=1, buff_size=2 ** 24
        )
        self.depth_sub = rospy.Subscriber(
            depth_topic, Image, self._depth_callback, queue_size=5, buff_size=2 ** 24
        )
        self.info_sub = rospy.Subscriber(
            info_topic, CameraInfo, self._info_callback, queue_size=1
        )

    def _color_callback(self, message: Image) -> None:
        with self.condition:
            self.color = message
            self.color_sequence += 1
            self.condition.notify_all()

    def _depth_callback(self, message: Image) -> None:
        with self.condition:
            self.depths.append(message)
            self.condition.notify_all()

    def _info_callback(self, message: CameraInfo) -> None:
        with self.condition:
            self.info = message
            self.condition.notify_all()

    def next_bundle(
        self, previous_sequence: int, deadline: float, max_sync_delta: float
    ) -> Tuple[int, Image, Image, CameraInfo, float]:
        with self.condition:
            while not rospy.is_shutdown():
                if (
                    self.color_sequence != previous_sequence
                    and self.color is not None
                    and self.depths
                    and self.info is not None
                ):
                    color = self.color
                    color_stamp = stamp_seconds(color)
                    depth = min(
                        self.depths,
                        key=lambda item: abs(stamp_seconds(item) - color_stamp),
                    )
                    delta = abs(stamp_seconds(depth) - color_stamp)
                    if delta <= max_sync_delta:
                        return self.color_sequence, color, depth, self.info, delta
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "timed out waiting for synchronized color, aligned depth, "
                        "and camera_info"
                    )
                self.condition.wait(min(remaining, 0.1))
        raise RuntimeError("ROS stopped while waiting for D435 data")


class TargetAcquirer:
    def __init__(self, args: argparse.Namespace, collector: SensorCollector) -> None:
        self.args = args
        self.collector = collector
        self.model = YOLO(str(args.model))
        self.debug_pub = rospy.Publisher("~debug_image", Image, queue_size=1, latch=True)

    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        prediction = self.model.predict(
            image,
            imgsz=self.args.imgsz,
            conf=self.args.confidence,
            iou=self.args.iou,
            device=self.args.device,
            max_det=self.args.max_det,
            verbose=False,
        )[0]
        if prediction.obb is None:
            return []
        corners = prediction.obb.xyxyxyxy.cpu().numpy()
        boxes = prediction.obb.xywhr.cpu().numpy()
        scores = prediction.obb.conf.cpu().numpy()
        return [
            {
                "center_px": np.asarray(box[:2], dtype=np.float64),
                "corners_px": np.asarray(points, dtype=np.float64),
                "confidence": float(score),
            }
            for points, box, score in zip(corners, boxes, scores)
        ]

    def draw(self, image: np.ndarray, header: Any, objects: List[Dict[str, Any]]) -> None:
        debug = image.copy()
        for index, item in enumerate(objects):
            selected = index == 0
            color = (0, 0, 255) if selected else (0, 255, 0)
            points = np.rint(item["corners_px"]).astype(np.int32)
            cv2.polylines(debug, [points], True, color, 4 if selected else 2)
            center = np.rint(item["center_px"]).astype(int)
            cv2.circle(debug, tuple(center), 6, color, -1)
            cv2.putText(
                debug,
                "SELECTED {:.3f}".format(item["confidence"]) if selected else "{:.3f}".format(item["confidence"]),
                tuple(points[0] + np.array([0, -8])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
        self.debug_pub.publish(bgr_to_image_message(debug, header))

    def acquire(self) -> Dict[str, Any]:
        deadline = time.monotonic() + self.args.input_timeout
        previous_sequence = -1
        last_capture_stamp: Optional[float] = None
        tracked_center: Optional[np.ndarray] = None
        samples: List[Dict[str, Any]] = []
        while not rospy.is_shutdown() and len(samples) < self.args.capture_frames:
            sequence, color_msg, depth_msg, info, sync_delta = self.collector.next_bundle(
                previous_sequence, deadline, self.args.max_sync_delta
            )
            previous_sequence = sequence
            capture_stamp = stamp_seconds(color_msg)
            if capture_stamp <= 0.0:
                capture_stamp = time.monotonic()
            if self.args.capture_fps > 0.0 and last_capture_stamp is not None:
                minimum_interval = 1.0 / self.args.capture_fps
                elapsed = capture_stamp - last_capture_stamp
                if 0.0 <= elapsed < minimum_interval:
                    continue
            image = color_message_to_bgr(color_msg)
            objects = sorted(
                self.infer(image), key=lambda item: item["confidence"], reverse=True
            )
            self.draw(image, color_msg.header, objects)
            if not objects:
                samples.clear()
                tracked_center = None
                rospy.logwarn_throttle(2.0, "No white rectangle detected")
                continue
            selected = objects[0]
            center = selected["center_px"]
            if tracked_center is not None:
                movement = float(np.linalg.norm(center - tracked_center))
                if movement > self.args.stability_px:
                    samples.clear()
                    rospy.logwarn("Target moved %.2f px; restarting stable samples", movement)
            tracked_center = center.copy()
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
                point = deproject_pixel(center[0], center[1], depth, info)
                edge_start, edge_end = long_edge_pixels(selected["corners_px"])
                edge_start_3d = deproject_pixel(edge_start[0], edge_start[1], depth, info)
                edge_end_3d = deproject_pixel(edge_end[0], edge_end[1], depth, info)
                edge_direction = normalize(edge_end_3d - edge_start_3d, "OBB long edge")
            except ValueError as error:
                samples.clear()
                rospy.logwarn_throttle(1.0, "Rejected target geometry: %s", error)
                continue
            if samples and float(np.dot(edge_direction, samples[-1]["edge_camera"])) < 0.0:
                edge_direction = -edge_direction
            samples.append(
                {
                    "center_px": center.copy(),
                    "point_camera": point,
                    "edge_camera": edge_direction,
                    "confidence": selected["confidence"],
                    "valid_depth": valid_count,
                    "sync_delta": sync_delta,
                }
            )
            last_capture_stamp = capture_stamp
            rospy.loginfo(
                "Captured target frame %d/%d center=[%.1f, %.1f] "
                "depth=%.4f confidence=%.3f",
                len(samples), self.args.capture_frames, center[0], center[1], depth,
                selected["confidence"],
            )
        if len(samples) < self.args.capture_frames:
            raise RuntimeError(
                "ROS stopped after {}/{} target frames".format(
                    len(samples), self.args.capture_frames
                )
            )
        return {
            # Median is more robust than mean when one YOLO box or one depth
            # patch is slightly displaced while the robot is stationary.
            "center_px": np.median(
                [item["center_px"] for item in samples], axis=0
            ),
            "point_camera": np.median(
                [item["point_camera"] for item in samples], axis=0
            ),
            "edge_camera": normalize(
                np.mean([item["edge_camera"] for item in samples], axis=0),
                "averaged OBB long edge",
            ),
            "confidence": float(np.mean([item["confidence"] for item in samples])),
            "captured_frames": len(samples),
        }


ROBOT_MODE_RUNNING = 7
SAFETY_MODE_NORMAL = 1


def verify_robot_ready(robot: UR_BASE) -> None:
    if robot.rtde_r is None:
        raise RuntimeError("RTDE receive interface is unavailable")
    robot_mode = int(robot.rtde_r.getRobotMode())
    safety_mode = int(robot.rtde_r.getSafetyMode())
    print("robot mode={}, safety mode={}".format(robot_mode, safety_mode))
    if safety_mode != SAFETY_MODE_NORMAL:
        raise RuntimeError("robot safety mode is not NORMAL(1)")
    if robot_mode != ROBOT_MODE_RUNNING:
        raise RuntimeError("robot mode is not RUNNING(7)")


def require_success(name: str, result: Any) -> None:
    print("{}: {}".format(name, result))
    if not result.success:
        raise RuntimeError("{} failed: {}".format(name, result.message))


def validate_workspace(point: np.ndarray, low: np.ndarray, high: np.ndarray, name: str) -> None:
    if np.any(point < low) or np.any(point > high):
        raise ValueError(
            "{} {} is outside workspace [{}, {}]".format(
                name, point.round(6).tolist(), low.tolist(), high.tolist()
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect a white rectangle and grasp it with aligned OBB orientation."
    )
    parser.add_argument("--host", default="192.168.1.5")
    parser.add_argument("--gripper-port", type=int, default=63352)
    parser.add_argument(
        "--observation-joints",
        nargs=6,
        type=float,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="measured joint angles in radians for the fixed observation pose; bypasses TCP IK",
    )
    parser.add_argument(
        "--tcp-offset",
        nargs=6,
        type=float,
        default=(0.0, 0.0, 0.135, 0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="active flange-to-TCP pose used only when it cannot be queried, metres/radians",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--handeye", type=Path, default=DEFAULT_HANDEYE)
    parser.add_argument("--color-topic", default="/d435/color/image_raw")
    parser.add_argument("--depth-topic", default="/d435/aligned_depth_to_color/image_raw")
    parser.add_argument("--camera-info-topic", default="/d435/color/camera_info")
    parser.add_argument(
        "--d435-color-offset",
        type=float,
        default=DEFAULT_D435_DEPTH_TO_COLOR_OFFSET_M,
        help="nominal d435_link-to-color-frame Y offset used when RealSense TF is absent",
    )
    parser.add_argument(
        "--require-camera-tf",
        action="store_true",
        help="fail instead of using the nominal D435 internal transform when TF is absent",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-det", type=int, default=10)
    parser.add_argument(
        "--capture-frames", "--stable-samples",
        dest="capture_frames", type=int, default=12,
        help=(
            "number of valid stable RGB-D detections fused before movement; "
            "--stable-samples is retained as a compatibility alias (default: 12)"
        ),
    )
    parser.add_argument(
        "--capture-fps", type=float, default=5.0,
        help=(
            "maximum accepted frame rate during target capture in Hz; "
            "use 0 for every camera frame (default: 5.0)"
        ),
    )
    parser.add_argument("--stability-px", type=float, default=6.0)
    parser.add_argument("--input-timeout", type=float, default=30.0)
    parser.add_argument("--max-sync-delta", type=float, default=0.10)
    parser.add_argument("--depth-radius", type=int, default=5)
    parser.add_argument("--min-depth", type=float, default=0.10)
    parser.add_argument("--max-depth", type=float, default=1.50)
    parser.add_argument("--pregrasp-distance", type=float, default=0.10)
    parser.add_argument("--grasp-distance", type=float, default=0.05)
    parser.add_argument(
        "--lift-distance", type=float, default=0.13,
        help="post-grasp MoveL distance along base +Z in metres (default: 0.13)",
    )
    parser.add_argument(
        "--final-pose", nargs=6, type=float,
        default=DEFAULT_FINAL_POSE.tolist(),
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="final MoveJ/IK TCP pose in base frame, metres/radians",
    )
    parser.add_argument(
        "--movej-speed", type=float, default=0.12,
        help="speed shared by every MoveJ/IK and MoveJ command (default: 0.12)",
    )
    parser.add_argument(
        "--movej-acc", type=float, default=0.20,
        help="acceleration shared by every MoveJ/IK and MoveJ command (default: 0.20)",
    )
    parser.add_argument(
        "--movel-speed", type=float, default=0.02,
        help="speed shared by every MoveL command in m/s (default: 0.02)",
    )
    parser.add_argument(
        "--movel-acc", type=float, default=0.05,
        help="acceleration shared by every MoveL command in m/s^2 (default: 0.05)",
    )
    parser.add_argument("--motion-timeout", type=float, default=30.0)
    parser.add_argument(
        "--workspace",
        nargs=6,
        type=float,
        default=(-0.45, 0.45, 0.05, 0.75, 0.03, 0.65),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="allowed UR controller Base XYZ bounds in metres",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--full-auto", "--full_auto", dest="full_auto", action="store_true",
        help="execute the complete sequence without waiting for Enter confirmations",
    )
    parser.add_argument(
        "--hold-preview",
        action="store_true",
        help="in dry-run mode, keep the ROS debug/result topics alive until Ctrl+C",
    )
    args = parser.parse_args(rospy.myargv()[1:])

    if not args.model.is_file():
        parser.error("model does not exist: {}".format(args.model))
    if not args.handeye.is_file():
        parser.error("hand-eye calibration does not exist: {}".format(args.handeye))
    if args.observation_joints is not None and not np.all(
        np.isfinite(np.asarray(args.observation_joints, dtype=np.float64))
    ):
        parser.error("--observation-joints must contain six finite values")
    if not np.all(np.isfinite(np.asarray(args.tcp_offset, dtype=np.float64))):
        parser.error("--tcp-offset must contain six finite values")
    if not 0.0 < args.confidence <= 1.0 or not 0.0 < args.iou <= 1.0:
        parser.error("--confidence and --iou must be in (0, 1]")
    if args.imgsz <= 0 or args.max_det <= 0 or args.capture_frames <= 0:
        parser.error("--imgsz, --max-det and --capture-frames must be positive")
    if not math.isfinite(args.capture_fps) or args.capture_fps < 0.0:
        parser.error("--capture-fps must be finite and >= 0")
    if args.depth_radius < 0 or args.stability_px <= 0.0:
        parser.error("--depth-radius must be >= 0 and --stability-px > 0")
    if not 0.0 < args.min_depth < args.max_depth:
        parser.error("require 0 < --min-depth < --max-depth")
    if not 0.0 < args.grasp_distance < args.pregrasp_distance:
        parser.error("require 0 < --grasp-distance < --pregrasp-distance")
    if not math.isfinite(args.lift_distance) or args.lift_distance <= 0.0:
        parser.error("--lift-distance must be a positive finite value")
    if not np.all(np.isfinite(np.asarray(args.final_pose, dtype=np.float64))):
        parser.error("--final-pose must contain six finite values")
    if not math.isfinite(args.d435_color_offset) or abs(args.d435_color_offset) > 0.10:
        parser.error("--d435-color-offset must be finite and within 0.10 m")
    for name in (
        "input_timeout", "max_sync_delta", "movej_speed", "movej_acc",
        "movel_speed", "movel_acc", "motion_timeout",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0.0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    bounds = np.asarray(args.workspace, dtype=np.float64)
    low, high = bounds[[0, 2, 4]], bounds[[1, 3, 5]]
    if not np.all(np.isfinite(bounds)) or np.any(low >= high):
        parser.error("invalid --workspace bounds")
    return args


def main() -> None:
    args = parse_args()
    rospy.init_node("white_rectangle_pick")
    collector = SensorCollector(
        args.color_topic, args.depth_topic, args.camera_info_topic
    )
    acquirer = TargetAcquirer(args, collector)
    result_pub = rospy.Publisher("~result", String, queue_size=1, latch=True)
    tool_from_d435 = load_tool_from_d435(args.handeye)
    handeye_mtime = datetime.fromtimestamp(
        args.handeye.stat().st_mtime
    ).astimezone().isoformat(timespec="seconds")
    rospy.loginfo(
        "Loaded hand-eye calibration %s (modified %s); translation [m]=%s",
        args.handeye,
        handeye_mtime,
        np.array2string(tool_from_d435[:3, 3], precision=6),
    )

    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    tf_listener = tf2_ros.TransformListener(tf_buffer)  # noqa: F841
    rospy.loginfo("Looking up d435_link <- d435_color_optical_frame TF")
    try:
        internal_tf = tf_buffer.lookup_transform(
            "d435_link", "d435_color_optical_frame", rospy.Time(0), rospy.Duration(3.0)
        )
        d435_from_optical = transform_message_to_matrix(internal_tf.transform)
        rospy.loginfo("Using device-published D435 internal optical TF")
    except (
        tf2_ros.LookupException,
        tf2_ros.ConnectivityException,
        tf2_ros.ExtrapolationException,
    ) as error:
        if args.require_camera_tf:
            raise RuntimeError("D435 internal optical TF is unavailable: {}".format(error))
        d435_from_optical = nominal_d435_from_color_optical(args.d435_color_offset)
        rospy.logwarn(
            "D435 internal optical TF is unavailable (%s). Using nominal "
            "d435_link <- color_optical transform with color Y offset %.6f m.",
            error,
            args.d435_color_offset,
        )

    connect_control = bool(args.execute)
    gripper_port = args.gripper_port if args.execute else None
    with UR_BASE(
        args.host, gripper_port=gripper_port, connect_control=connect_control
    ) as robot:
        verify_robot_ready(robot)
        if args.execute:
            if not args.full_auto:
                input(
                    "Press Enter to move to the observation pose and open the gripper "
                    "(Ctrl+C to cancel): "
                )
            if args.observation_joints is None:
                observation_motion = robot.move_j_ik(
                    OBSERVATION_POSE.tolist(),
                    args.movej_speed,
                    args.movej_acc,
                    args.motion_timeout,
                )
                observation_motion_name = "moveJ/IK observation pose"
            else:
                observation_motion = robot.move_j(
                    args.observation_joints,
                    args.movej_speed,
                    args.movej_acc,
                    args.motion_timeout,
                )
                observation_motion_name = "moveJ observation joints"
            require_success(observation_motion_name, observation_motion)
            require_success(
                "open gripper",
                robot.open_gripper(speed=255, force=255, timeout_s=5.0),
            )
            time.sleep(1.0)
        else:
            print("DRY RUN: robot will not move; using the current measured TCP pose.")

        observation_actual = np.asarray(robot.get_tcp_pose_or_raise(), dtype=np.float64)
        observation_error_mm = float(
            np.linalg.norm(observation_actual[:3] - OBSERVATION_POSE[:3]) * 1000.0
        )
        print("actual observation TCP: {}".format(observation_actual.round(9).tolist()))
        print("observation position error: {:.3f} mm".format(observation_error_mm))

        active_tcp_offset = np.asarray(args.tcp_offset, dtype=np.float64)
        if robot.rtde_c is not None:
            active_tcp_offset = np.asarray(
                robot.rtde_c.getTCPOffset(), dtype=np.float64
            )
        if active_tcp_offset.shape != (6,) or not np.all(np.isfinite(active_tcp_offset)):
            raise RuntimeError("controller returned an invalid active TCP offset")
        base_from_active_tcp = pose_to_matrix(observation_actual)
        tool_from_active_tcp = pose_to_matrix(active_tcp_offset)
        base_from_tool = base_from_active_tcp @ np.linalg.inv(tool_from_active_tcp)
        base_from_optical = base_from_tool @ tool_from_d435 @ d435_from_optical
        print("active TCP offset [m,rad]: {}".format(
            active_tcp_offset.round(9).tolist()
        ))
        print("UR-base <- color-optical matrix:\n{}".format(
            np.array2string(base_from_optical, precision=8, suppress_small=True)
        ))

        rospy.loginfo(
            "Capturing %d stable target frames at up to %.3f Hz; view "
            "/white_rectangle_pick/debug_image",
            args.capture_frames,
            args.capture_fps,
        )
        target = acquirer.acquire()
        point_camera_h = np.append(target["point_camera"], 1.0)
        target_base = (base_from_optical @ point_camera_h)[:3]
        edge_base = normalize(
            base_from_optical[:3, :3] @ target["edge_camera"],
            "OBB long edge in UR base",
        )
        camera_origin_base = base_from_optical[:3, 3]
        camera_approach_axis = normalize(
            target_base - camera_origin_base, "camera-to-target axis"
        )
        target_tcp_rotation, selected_finger_axis, approach_axis = desired_tcp_rotation(
            camera_approach_axis, base_from_active_tcp[:3, :3]
        )
        pregrasp_xyz = target_base - approach_axis * args.pregrasp_distance
        grasp_xyz = target_base - approach_axis * args.grasp_distance
        target_rotvec = matrix_to_rotvec(target_tcp_rotation)
        pregrasp_pose = pregrasp_xyz.tolist() + target_rotvec.tolist()
        grasp_pose = grasp_xyz.tolist() + target_rotvec.tolist()
        lift_pose = np.asarray(grasp_pose, dtype=np.float64)
        lift_pose[2] += args.lift_distance
        final_pose = np.asarray(args.final_pose, dtype=np.float64)

        bounds = np.asarray(args.workspace, dtype=np.float64)
        low, high = bounds[[0, 2, 4]], bounds[[1, 3, 5]]
        validate_workspace(pregrasp_xyz, low, high, "pregrasp point")
        validate_workspace(grasp_xyz, low, high, "grasp point")
        validate_workspace(lift_pose[:3], low, high, "post-grasp lift point")
        validate_workspace(final_pose[:3], low, high, "final MoveJ point")

        payload = {
            "confidence": round(target["confidence"], 5),
            "captured_frames": target["captured_frames"],
            "capture_fps": args.capture_fps,
            "center_px": target["center_px"].round(3).tolist(),
            "camera_point_m": target["point_camera"].round(6).tolist(),
            "target_ur_base_m": target_base.round(6).tolist(),
            "long_edge_ur_base": edge_base.round(6).tolist(),
            "selected_finger_axis_ur_base": selected_finger_axis.round(6).tolist(),
            "camera_approach_axis_ur_base": camera_approach_axis.round(6).tolist(),
            "approach_axis_ur_base": approach_axis.round(6).tolist(),
            "pregrasp_pose": np.asarray(pregrasp_pose).round(6).tolist(),
            "grasp_pose": np.asarray(grasp_pose).round(6).tolist(),
            "lift_distance_m": args.lift_distance,
            "planned_lift_pose": lift_pose.round(6).tolist(),
            "final_pose": final_pose.round(6).tolist(),
        }
        result_pub.publish(String(data=json.dumps(payload)))
        print("\n" + "=" * 76)
        print("YOLO confidence: {:.4f}".format(target["confidence"]))
        print("Captured frames: {} at up to {:.3f} Hz".format(
            target["captured_frames"], args.capture_fps
        ))
        print("OBB center [u,v] px: {}".format(target["center_px"].round(3).tolist()))
        print("camera target XYZ [m]: {}".format(target["point_camera"].round(6).tolist()))
        print("UR-base target XYZ [m]: {}".format(target_base.round(6).tolist()))
        print("OBB long edge in UR base: {}".format(edge_base.round(6).tolist()))
        print("selected finger line in UR base: {}".format(
            selected_finger_axis.round(6).tolist()
        ))
        print("camera-to-target axis in UR base: {}".format(
            camera_approach_axis.round(6).tolist()
        ))
        print("approach axis in UR base: {}".format(approach_axis.round(6).tolist()))
        print("pregrasp {:.1f} cm pose: {}".format(
            args.pregrasp_distance * 100.0,
            np.asarray(pregrasp_pose).round(6).tolist(),
        ))
        print("grasp {:.1f} cm pose: {}".format(
            args.grasp_distance * 100.0,
            np.asarray(grasp_pose).round(6).tolist(),
        ))
        print("post-grasp lift +{:.1f} cm pose: {}".format(
            args.lift_distance * 100.0, lift_pose.round(6).tolist()
        ))
        print("final MoveJ/IK pose: {}".format(final_pose.round(6).tolist()))
        print("=" * 76)

        if not args.execute:
            print("DRY RUN complete: no movement and no gripper command were sent.")
            print("Inspect /white_rectangle_pick/debug_image and the printed coordinates.")
            if args.hold_preview:
                print("Preview is being held; press Ctrl+C to exit.")
                rospy.spin()
            return

        if not args.full_auto:
            input(
                "Press Enter to execute grasp, lift, and final MoveJ sequence "
                "(Ctrl+C to cancel): "
            )
        require_success(
            "moveJ pregrasp {:.1f} cm".format(args.pregrasp_distance * 100.0),
            robot.move_j_ik(
                pregrasp_pose,
                args.movej_speed,
                args.movej_acc,
                args.motion_timeout,
            ),
        )
        require_success(
            "moveL grasp {:.1f} cm".format(args.grasp_distance * 100.0),
            robot.move_l(
                grasp_pose,
                args.movel_speed,
                args.movel_acc,
                args.motion_timeout,
            ),
        )
        require_success(
            "close gripper",
            robot.close_gripper(speed=255, force=255, timeout_s=5.0),
        )
        actual_grasp_pose = np.asarray(robot.get_tcp_pose_or_raise(), dtype=np.float64)
        actual_lift_pose = actual_grasp_pose.copy()
        actual_lift_pose[2] += args.lift_distance
        validate_workspace(actual_lift_pose[:3], low, high, "actual post-grasp lift point")
        require_success(
            "moveL lift +{:.1f} cm in base Z".format(args.lift_distance * 100.0),
            robot.move_l(
                actual_lift_pose.tolist(),
                args.movel_speed,
                args.movel_acc,
                args.motion_timeout,
            ),
        )
        require_success(
            "moveJ/IK final pose",
            robot.move_j_ik(
                final_pose.tolist(),
                args.movej_speed,
                args.movej_acc,
                args.motion_timeout,
            ),
        )
        print("Grasp, lift, and final MoveJ sequence completed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
    except (RuntimeError, ValueError, FileNotFoundError, TimeoutError, yaml.YAMLError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
