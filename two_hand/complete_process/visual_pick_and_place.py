#!/usr/bin/env python3
"""YOLO white-workpiece pick and depth-difference red-board placement.

The gripper reference orientation was measured with the tool vertical and the
line between the two fingers parallel to the UR controller Base X axis. White
parts are grasped with that finger line parallel to the edge-refined,
inscribed rectangle's short edge.
At placement, the finger line is aligned with the long edge of the red depth
rectangle.

The program is a dry run unless --execute is supplied. At placement it moves
to the configured red-slot approach height, descends to the slot surface, and
then opens the gripper at the resulting release height.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")

import rospy
import tf2_ros
from sensor_msgs.msg import Image
from std_msgs.msg import String

from complete_process import detect_and_pick as vision
from complete_process.utils.base.ur_base import UR_BASE
from complete_process.utils.object_detective import (
    white_threshold_segmenter as white_edge,
)


DEFAULT_WHITE_MODEL = (
    PROJECT_ROOT / "complete_process/utils/object_detective/models/white_box.pt"
)
DEFAULT_DEPTH_DETECTOR = (
    PROJECT_ROOT
    / "complete_process/utils/object_detective/depth_background_rect_detector.py"
)

# Measured TCP orientation on 2026-08-21 with the tool vertical and the
# finger-to-finger line parallel to UR controller Base X.  At this reference
# pose the six joints are approximately [90, -90, 90, 270, -90, 270] degrees
# (angles are equivalent modulo 360 degrees).  The measured TCP position is
# deliberately not part of this orientation-only calibration.
GRIPPER_BASE_X_REFERENCE_ROTVEC = np.array(
    [2.204603, -2.213467, -0.020398], dtype=np.float64
)


def rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotvec_to_matrix(rotvec: Sequence[float]) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("rotation vector must contain three finite values")
    matrix, _ = cv2.Rodrigues(vector.reshape(3, 1))
    return matrix


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    vector, _ = cv2.Rodrigues(np.asarray(matrix, dtype=np.float64))
    return vector.reshape(3)


def rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def aligned_gripper_rotvec(
    target_angle_in_ur_base: float,
    nearby_rotvec: Sequence[float],
) -> np.ndarray:
    """Return the closer of the two 180-degree-equivalent gripper poses."""
    reference = rotvec_to_matrix(GRIPPER_BASE_X_REFERENCE_ROTVEC)
    nearby = rotvec_to_matrix(nearby_rotvec)
    candidates = [
        rotation_z(target_angle_in_ur_base) @ reference,
        rotation_z(target_angle_in_ur_base + math.pi) @ reference,
    ]
    selected = min(candidates, key=lambda item: rotation_distance(nearby, item))
    return matrix_to_rotvec(selected)


def pixel_ray_in_camera(u: float, v: float, camera_info: Any) -> np.ndarray:
    fx, fy = float(camera_info.K[0]), float(camera_info.K[4])
    cx, cy = float(camera_info.K[2]), float(camera_info.K[5])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("invalid color-camera intrinsics")
    return np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)


def pixel_to_base_plane(
    pixel: Sequence[float],
    plane_z: float,
    camera_info: Any,
    base_from_camera: np.ndarray,
) -> np.ndarray:
    """Intersect a color-camera pixel ray with a horizontal base_link plane."""
    origin = base_from_camera[:3, 3]
    ray = base_from_camera[:3, :3] @ pixel_ray_in_camera(
        float(pixel[0]), float(pixel[1]), camera_info
    )
    if abs(float(ray[2])) < 1e-9:
        raise ValueError("camera ray is parallel to the target plane")
    scale = (float(plane_z) - float(origin[2])) / float(ray[2])
    if scale <= 0.0:
        raise ValueError("target plane lies behind the camera")
    return origin + scale * ray


def obb_on_base_plane(
    target: Dict[str, Any],
    plane_z: float,
    camera_info: Any,
    base_from_camera: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return center, short/long unit axes and side lengths in base_link."""
    pixels = np.asarray(target["corners_px"], dtype=np.float64)
    if pixels.shape != (4, 2):
        raise ValueError("YOLO OBB must contain four image corners")
    corners = np.stack(
        [
            pixel_to_base_plane(pixel, plane_z, camera_info, base_from_camera)
            for pixel in pixels
        ]
    )
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges[:, :2], axis=1)
    if np.any(lengths < 1e-6):
        raise ValueError("YOLO OBB contains a degenerate edge")
    short_index = int(np.argmin(lengths))
    long_index = int(np.argmax(lengths))
    short_axis = edges[short_index].copy()
    long_axis = edges[long_index].copy()
    short_axis[2] = 0.0
    long_axis[2] = 0.0
    short_axis /= np.linalg.norm(short_axis)
    long_axis /= np.linalg.norm(long_axis)
    return (
        corners.mean(axis=0),
        short_axis,
        long_axis,
        float(lengths[short_index]),
        float(lengths[long_index]),
    )


def base_link_axis_angle_in_ur_base(axis: Sequence[float]) -> float:
    vector = np.asarray(axis, dtype=np.float64)
    # base_link -> UR controller base is Rz(pi): X and Y change sign.
    return math.atan2(-float(vector[1]), -float(vector[0]))


def ordered_fractional_points(
    center: Sequence[float],
    axis: Sequence[float],
    length: float,
    fractions: Sequence[float],
) -> List[np.ndarray]:
    """Points along an axis starting at the endpoint with smaller base_link X."""
    center_xyz = np.asarray(center, dtype=np.float64)
    unit_axis = np.asarray(axis, dtype=np.float64)
    first = center_xyz - 0.5 * float(length) * unit_axis
    second = center_xyz + 0.5 * float(length) * unit_axis
    # X is the required primary ordering. Y makes the near-vertical X tie
    # deterministic without changing the physical definition in normal cases.
    if (second[0], second[1]) < (first[0], first[1]):
        first, second = second, first
    direction = second - first
    return [first + float(fraction) * direction for fraction in fractions]


def fractional_point_from_nearer_base_origin(
    center: Sequence[float],
    axis: Sequence[float],
    length: float,
    fraction: float,
) -> np.ndarray:
    """Measure a fraction from the axis endpoint nearer the base_link origin."""
    center_xyz = np.asarray(center, dtype=np.float64)
    unit_axis = np.asarray(axis, dtype=np.float64)
    first = center_xyz - 0.5 * float(length) * unit_axis
    second = center_xyz + 0.5 * float(length) * unit_axis
    if np.linalg.norm(second) < np.linalg.norm(first):
        first, second = second, first
    return first + float(fraction) * (second - first)


def fractional_point_from_farther_base_origin(
    center: Sequence[float],
    axis: Sequence[float],
    length: float,
    fraction: float,
) -> np.ndarray:
    """Measure a fraction from the axis endpoint farther from base_link origin."""
    center_xyz = np.asarray(center, dtype=np.float64)
    unit_axis = np.asarray(axis, dtype=np.float64)
    first = center_xyz - 0.5 * float(length) * unit_axis
    second = center_xyz + 0.5 * float(length) * unit_axis
    if np.linalg.norm(first) < np.linalg.norm(second):
        first, second = second, first
    return first + float(fraction) * (second - first)


def make_ur_pose(base_link_xyz: Sequence[float], rotvec: Sequence[float]) -> List[float]:
    ur_xyz = vision.base_link_point_to_ur_base(base_link_xyz)
    return ur_xyz.tolist() + np.asarray(rotvec, dtype=np.float64).tolist()


def validate_workspace(
    point: Sequence[float], low: np.ndarray, high: np.ndarray, name: str
) -> None:
    xyz = np.asarray(point, dtype=np.float64)
    if np.any(xyz < low) or np.any(xyz > high):
        raise ValueError(
            "{} {} is outside base_link workspace [{}, {}]".format(
                name, xyz.round(6).tolist(), low.tolist(), high.tolist()
            )
        )


def wait_for_base_camera_matrix(args: argparse.Namespace) -> np.ndarray:
    buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    listener = tf2_ros.TransformListener(buffer)  # noqa: F841
    base_from_camera_link = vision.calibration_to_matrix(
        args.handeye_translation, args.handeye_quaternion
    )
    uses_default = np.allclose(
        np.asarray(args.handeye_translation, dtype=np.float64),
        np.asarray(vision.DEFAULT_HAND_EYE_TRANSLATION, dtype=np.float64),
    ) and np.allclose(
        np.asarray(args.handeye_quaternion, dtype=np.float64),
        np.asarray(vision.DEFAULT_HAND_EYE_QUATERNION, dtype=np.float64),
    )
    calibration_source = (
        vision.DEFAULT_HAND_EYE_SOURCE if uses_default else "command-line values"
    )
    rospy.loginfo(
        "Hand-eye base_link <- %s: source=%s translation=%s quaternion_xyzw=%s",
        args.handeye_camera_frame,
        calibration_source,
        np.asarray(args.handeye_translation).round(9).tolist(),
        np.asarray(args.handeye_quaternion).round(9).tolist(),
    )
    try:
        internal = buffer.lookup_transform(
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
            "RealSense TF {} <- {} unavailable: {}".format(
                args.handeye_camera_frame, args.camera_frame, error
            )
        )
    return base_from_camera_link @ vision.transform_to_matrix(internal.transform)


def acquire_with_model(
    acquirer: vision.YoloTargetAcquirer,
    model_path: Path,
    label: str,
) -> Dict[str, Any]:
    rospy.loginfo("Loading %s model: %s", label, model_path)
    acquirer.model = YOLO(str(model_path))
    rospy.loginfo(
        "Waiting to capture a fixed batch of %d valid %s frames...",
        acquirer.args.stable_samples,
        label,
    )
    target = acquirer.acquire()
    rospy.loginfo(
        "%s target acquired: confidence=%.4f center=[%.2f, %.2f]",
        label,
        target["confidence"],
        target["center_px"][0],
        target["center_px"][1],
    )
    return target


def white_edge_config(args: argparse.Namespace) -> argparse.Namespace:
    """Translate pick-flow arguments to the reusable edge segmenter config."""
    return argparse.Namespace(
        search_scale=args.white_search_scale,
        edge_blur_kernel=int(args.white_edge_blur_kernel) | 1,
        canny_low=args.white_canny_low,
        canny_high=args.white_canny_high,
        edge_close_kernel=args.white_edge_close_kernel,
        edge_close_iterations=args.white_edge_close_iterations,
        min_area=args.white_edge_min_area,
        close_kernel=args.white_mask_close_kernel,
        close_iterations=args.white_mask_close_iterations,
        open_kernel=args.white_mask_open_kernel,
        contour_epsilon=args.white_contour_epsilon,
        inscribed_angle_range_deg=args.white_inscribed_angle_range_deg,
        inscribed_angle_step_deg=args.white_inscribed_angle_step_deg,
        inscribed_margin_px=args.white_inscribed_margin_px,
    )


class WhiteEdgeTargetAcquirer(vision.YoloTargetAcquirer):
    """YOLO tracker whose white geometry is replaced by an inner rectangle."""

    def __init__(
        self, args: argparse.Namespace, collector: vision.SensorCollector
    ) -> None:
        super().__init__(args, collector)
        self.edge_config = white_edge_config(args)

    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        yolo_objects = super().infer(image)
        refined: List[Dict[str, Any]] = []
        for item in yolo_objects:
            yolo_corners = np.asarray(item["corners_px"], dtype=np.float64)
            try:
                (
                    contour,
                    rectangle,
                    _,
                    _,
                    rectangle_area,
                    rectangle_angle,
                ) = white_edge.segment_white(
                    image, yolo_corners, self.edge_config
                )
            except ValueError as error:
                rospy.logwarn_throttle(
                    1.0,
                    "Rejected YOLO white box because edge refinement failed: %s",
                    error,
                )
                continue
            rectangle_float = rectangle.astype(np.float64)
            edges = np.roll(rectangle_float, -1, axis=0) - rectangle_float
            lengths = np.linalg.norm(edges, axis=1)
            refined_item = dict(item)
            refined_item["yolo_corners_px"] = yolo_corners
            refined_item["detailed_contour_px"] = contour.reshape(-1, 2).astype(
                np.float64
            )
            refined_item["corners_px"] = rectangle_float
            refined_item["center_px"] = np.mean(rectangle_float, axis=0).tolist()
            refined_item["size_px"] = [
                float(np.min(lengths)),
                float(np.max(lengths)),
            ]
            refined_item["angle_deg"] = float(rectangle_angle)
            refined_item["inscribed_rectangle_area_px"] = float(rectangle_area)
            refined.append(refined_item)
        return refined

    def draw_and_publish(
        self,
        image: np.ndarray,
        header: Any,
        objects: List[Dict[str, Any]],
        selected: Any,
    ) -> None:
        debug = image.copy()
        for item in objects:
            is_selected = item is selected
            yolo = np.rint(item["yolo_corners_px"]).astype(np.int32)
            contour = np.rint(item["detailed_contour_px"]).astype(np.int32)
            rectangle = np.rint(item["corners_px"]).astype(np.int32)
            cv2.polylines(debug, [yolo], True, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.polylines(debug, [contour], True, (255, 0, 255), 3, cv2.LINE_AA)
            cv2.polylines(
                debug,
                [rectangle],
                True,
                (255, 0, 0) if not is_selected else (0, 0, 255),
                4 if is_selected else 3,
                cv2.LINE_AA,
            )
            center = tuple(np.rint(item["center_px"]).astype(np.int32))
            cv2.drawMarker(
                debug,
                center,
                (0, 0, 255) if is_selected else (255, 0, 0),
                cv2.MARKER_CROSS,
                24,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug,
                "{}#{:d} {:.2f}".format(
                    "SELECTED " if is_selected else "",
                    int(item["id"]),
                    float(item["confidence"]),
                ),
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255) if is_selected else (255, 0, 255),
                2,
            )
        cv2.putText(
            debug,
            "yellow=YOLO magenta=edge blue/red=inscribed rectangle",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        self.debug_pub.publish(vision.bgr_to_image_message(debug, header))

    def acquire(self) -> Dict[str, Any]:
        """Capture exactly N valid frames without resetting on coordinate jumps."""
        discovery_deadline = time.monotonic() + self.args.input_timeout
        capture_deadline = None
        previous_sequence = -1
        tracked_center = None
        samples: List[Dict[str, Any]] = []

        while not rospy.is_shutdown() and len(samples) < self.args.stable_samples:
            active_deadline = (
                discovery_deadline if not samples else float(capture_deadline)
            )
            if time.monotonic() >= active_deadline:
                if not samples:
                    raise TimeoutError(
                        "no white workpiece detected within {:.1f} s".format(
                            self.args.input_timeout
                        )
                    )
                raise RuntimeError(
                    "white workpiece disappeared before the fixed batch was "
                    "complete: captured {}/{} valid frames".format(
                        len(samples), self.args.stable_samples
                    )
                )
            try:
                sequence, color_msg, depth_msg, info, sync_delta = (
                    self.collector.next_bundle(
                        previous_sequence,
                        active_deadline,
                        self.args.max_sync_delta,
                    )
                )
            except TimeoutError:
                if not samples:
                    raise TimeoutError(
                        "no white workpiece detected within {:.1f} s".format(
                            self.args.input_timeout
                        )
                    )
                raise RuntimeError(
                    "timed out after capturing {}/{} valid white frames".format(
                        len(samples), self.args.stable_samples
                    )
                )
            previous_sequence = sequence
            image = vision.color_message_to_bgr(color_msg)
            objects = self.infer(image)
            selected = None

            if objects:
                if tracked_center is None:
                    selected = random.choice(objects)
                    rospy.loginfo(
                        "Selected white target %d of %d at [%.2f, %.2f]",
                        int(selected["id"]) + 1,
                        len(objects),
                        selected["center_px"][0],
                        selected["center_px"][1],
                    )
                else:
                    selected = min(
                        objects,
                        key=lambda item: np.linalg.norm(
                            np.asarray(item["center_px"], dtype=np.float64)
                            - tracked_center
                        ),
                    )
                    association_distance = float(
                        np.linalg.norm(
                            np.asarray(selected["center_px"], dtype=np.float64)
                            - tracked_center
                        )
                    )
                    if association_distance > self.args.association_px:
                        rospy.logwarn_throttle(
                            1.0,
                            "Ignoring a different white object %.1f px from "
                            "the tracked batch target",
                            association_distance,
                        )
                        selected = None

            if selected is None:
                self.draw_and_publish(image, color_msg.header, objects, None)
                rospy.logwarn_throttle(
                    2.0,
                    "Waiting for a valid edge-refined white frame; batch remains %d/%d",
                    len(samples),
                    self.args.stable_samples,
                )
                continue

            center = np.asarray(selected["center_px"], dtype=np.float64)
            try:
                depth_m = vision.depth_message_to_meters(depth_msg)
                depth, valid_count = vision.robust_depth(
                    depth_m,
                    center[0],
                    center[1],
                    self.args.depth_radius,
                    self.args.min_depth,
                    self.args.max_depth,
                )
                camera_point = vision.deproject_pixel(
                    center[0], center[1], depth, info
                )
            except ValueError as error:
                rospy.logwarn_throttle(
                    1.0,
                    "Ignoring white frame with invalid center depth: %s",
                    error,
                )
                self.draw_and_publish(image, color_msg.header, objects, selected)
                continue

            sample = dict(selected)
            sample["camera_point_m"] = camera_point[:3]
            sample["depth_m"] = depth
            sample["depth_valid_count"] = valid_count
            sample["sync_delta_s"] = sync_delta
            samples.append(sample)
            # Track by the median of every captured center. An abnormal frame
            # stays in the fixed batch for later rejection but cannot drag the
            # association center toward another object.
            tracked_center = np.median(
                np.asarray([item["center_px"] for item in samples]), axis=0
            )
            if capture_deadline is None:
                capture_deadline = time.monotonic() + max(
                    30.0, 3.0 * float(self.args.stable_samples)
                )
            self.draw_and_publish(image, color_msg.header, objects, selected)
            rospy.loginfo(
                "Captured white coordinate frame %d/%d: center=[%.2f, %.2f], "
                "depth=%.4f m",
                len(samples),
                self.args.stable_samples,
                center[0],
                center[1],
                depth,
            )

        if len(samples) != self.args.stable_samples:
            raise RuntimeError(
                "ROS stopped after {}/{} white frames".format(
                    len(samples), self.args.stable_samples
                )
            )

        result = dict(samples[-1])
        result["center_px"] = np.mean(
            [sample["center_px"] for sample in samples], axis=0
        ).tolist()
        result["camera_point_m"] = np.mean(
            [sample["camera_point_m"] for sample in samples], axis=0
        ).tolist()
        result["depth_m"] = float(np.median([sample["depth_m"] for sample in samples]))
        result["confidence"] = float(
            np.mean([sample["confidence"] for sample in samples])
        )
        result["depth_valid_count"] = int(
            np.mean([sample["depth_valid_count"] for sample in samples])
        )
        result["_captured_samples"] = samples
        result["captured_samples"] = len(samples)
        return result


def acquire_depth_red_target(
    args: argparse.Namespace,
    collector: vision.SensorCollector,
) -> Dict[str, Any]:
    """Wait for the depth detector and attach synchronized metric depth."""
    deadline = time.monotonic() + args.depth_detection_timeout
    selected = None
    while not rospy.is_shutdown():
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(
                "timed out waiting for a red rectangle on {}".format(
                    args.depth_result_topic
                )
            )
        try:
            message = rospy.wait_for_message(
                args.depth_result_topic, String, timeout=min(remaining, 1.0)
            )
        except rospy.ROSException:
            continue
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError) as error:
            rospy.logwarn_throttle(2.0, "Invalid depth detector JSON: %s", error)
            continue
        if not payload.get("background_ready", False):
            rospy.logwarn_throttle(2.0, "Depth background is not ready; press B")
            continue
        if str(payload.get("target_color", "")).lower() != "red":
            rospy.logwarn_throttle(
                2.0, "Depth detector target_color is not red: %s",
                payload.get("target_color"),
            )
            continue
        objects = payload.get("objects") or []
        if not objects:
            rospy.logwarn_throttle(2.0, "No red depth rectangle detected")
            continue
        selected = max(objects, key=lambda item: float(item.get("area_px", 0.0)))
        break

    if selected is None:
        raise RuntimeError(
            "ROS was shut down before red depth detection completed; do not "
            "press Ctrl+C during the pick-and-place sequence"
        )

    with collector.condition:
        previous_sequence = collector.color_sequence
    _, color_message, depth_message, camera_info, sync_delta = collector.next_bundle(
        previous_sequence,
        time.monotonic() + args.input_timeout,
        args.max_sync_delta,
    )
    center = np.asarray(selected["center_px"], dtype=np.float64)
    corners = np.asarray(selected["corners_px"], dtype=np.float64)
    if center.shape != (2,) or corners.shape != (4, 2):
        raise ValueError("depth rectangle has invalid center/corners")
    depth_m = vision.depth_message_to_meters(depth_message)
    depth, valid_count = vision.robust_depth(
        depth_m,
        float(center[0]),
        float(center[1]),
        args.depth_radius,
        args.min_depth,
        args.max_depth,
    )
    camera_point = vision.deproject_pixel(
        float(center[0]), float(center[1]), depth, camera_info
    )[:3]
    rospy.loginfo(
        "Red depth rectangle: area=%.1f center=[%.2f, %.2f] depth=%.4f "
        "valid=%d sync=%.4fs",
        float(selected.get("area_px", 0.0)), center[0], center[1], depth,
        valid_count, sync_delta,
    )
    return {
        "center_px": center.tolist(),
        "corners_px": corners.tolist(),
        "camera_point_m": camera_point.tolist(),
        "depth_m": depth,
        "area_px": float(selected.get("area_px", 0.0)),
        # Internal-only field used to draw the rectangle and robot targets on
        # the synchronized color image.  It is not included in JSON results.
        "color_message": color_message,
    }


def verify_depth_detector_ready(args: argparse.Namespace) -> None:
    """Fail before robot motion if the external depth detector is unavailable."""
    try:
        message = rospy.wait_for_message(
            args.depth_result_topic,
            String,
            timeout=args.depth_detection_timeout,
        )
    except rospy.ROSException as error:
        raise RuntimeError(
            "depth detector topic {} is unavailable: {}".format(
                args.depth_result_topic, error
            )
        )
    try:
        payload = json.loads(message.data)
    except (TypeError, ValueError) as error:
        raise RuntimeError("depth detector published invalid JSON: {}".format(error))
    if not payload.get("background_ready", False):
        raise RuntimeError(
            "depth detector background is not ready; remove the parts and press B"
        )
    if str(payload.get("target_color", "")).lower() != "red":
        raise RuntimeError(
            "depth detector must run with _target_color:=red (current: {!r})".format(
                payload.get("target_color")
            )
        )
    rospy.loginfo(
        "Depth detector ready: topic=%s target_color=red count=%d",
        args.depth_result_topic,
        int(payload.get("count", 0)),
    )


def stop_depth_detector(process: subprocess.Popen) -> None:
    """Stop only the detector process that this program started."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def ensure_depth_detector(args: argparse.Namespace) -> Any:
    """Reuse an external detector, or start a private one for this workflow."""
    published = {name for name, _ in rospy.get_published_topics()}
    if args.depth_result_topic in published:
        rospy.loginfo(
            "Using the existing depth detector: %s", args.depth_result_topic
        )
        verify_depth_detector_ready(args)
        return None

    if not args.auto_start_depth_detector:
        verify_depth_detector_ready(args)
        return None

    command = [
        str(args.depth_detector_python),
        "-u",
        str(DEFAULT_DEPTH_DETECTOR),
        "_color_topic:={}".format(args.color_topic),
        "_depth_topic:={}".format(args.depth_topic),
        "_target_color:=red",
        "_min_area:={}".format(args.red_min_area),
        "_collect_dataset:=false",
        "_show_gui:=false",
        "__name:=depth_background_rect_detector",
    ]
    rospy.loginfo("Depth detector is absent; starting it automatically")
    rospy.loginfo("Depth detector command: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=str(DEFAULT_DEPTH_DETECTOR.parent),
        # The detector is a background worker.  It must not compete with the
        # main workflow for the terminal used by input("... MOVE ...").
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        verify_depth_detector_ready(args)
    except Exception:
        exit_code = process.poll()
        stop_depth_detector(process)
        if exit_code is not None:
            rospy.logerr("Automatically started depth detector exited: %d", exit_code)
        raise

    # Both callbacks are intentional: rospy shutdown can happen while a robot
    # call is blocking, while atexit also covers ordinary Python exceptions.
    rospy.on_shutdown(lambda: stop_depth_detector(process))
    atexit.register(stop_depth_detector, process)
    return process


def base_surface_point(target: Dict[str, Any], matrix: np.ndarray) -> np.ndarray:
    camera_point = np.append(
        np.asarray(target["camera_point_m"], dtype=np.float64), 1.0
    )
    return (matrix @ camera_point)[:3]


def latest_camera_info(collector: vision.SensorCollector) -> Any:
    with collector.condition:
        info = collector.camera_info
    if info is None:
        raise RuntimeError("color camera_info is unavailable")
    return info


def print_pose(name: str, pose: Sequence[float]) -> None:
    print("{}: {}".format(name, [round(float(value), 6) for value in pose]))


def confirm(prompt: str, skip: bool) -> None:
    if skip:
        return
    if input(prompt).strip() != "MOVE":
        raise RuntimeError("operation cancelled by operator")


def run_motion(name: str, action: Any) -> None:
    if rospy.is_shutdown():
        raise RuntimeError(
            "ROS shutdown detected before {}; Ctrl+C cancels the remaining "
            "pick-and-place sequence".format(name)
        )
    result = action()
    print("{}: {}".format(name, result))
    if not result.success:
        raise RuntimeError("{} failed: {}".format(name, result.message))
    if rospy.is_shutdown():
        raise RuntimeError(
            "ROS shutdown detected during {}; remaining robot motions were "
            "cancelled".format(name)
        )


def add_white_edge_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--white-search-scale", type=float, default=1.08)
    parser.add_argument("--white-canny-low", type=int, default=35)
    parser.add_argument("--white-canny-high", type=int, default=110)
    parser.add_argument("--white-edge-blur-kernel", type=int, default=5)
    parser.add_argument("--white-edge-close-kernel", type=int, default=9)
    parser.add_argument("--white-edge-close-iterations", type=int, default=2)
    parser.add_argument("--white-edge-min-area", type=int, default=1200)
    parser.add_argument("--white-mask-close-kernel", type=int, default=11)
    parser.add_argument("--white-mask-close-iterations", type=int, default=2)
    parser.add_argument("--white-mask-open-kernel", type=int, default=3)
    parser.add_argument("--white-contour-epsilon", type=float, default=0.002)
    parser.add_argument("--white-inscribed-angle-range-deg", type=float, default=2.0)
    parser.add_argument("--white-inscribed-angle-step-deg", type=float, default=1.0)
    parser.add_argument("--white-inscribed-margin-px", type=int, default=3)
    parser.add_argument("--white-outlier-mad-scale", type=float, default=3.5)
    parser.add_argument("--white-outlier-min-mm", type=float, default=3.0)
    parser.add_argument("--white-min-inliers", type=int, default=6)


def validate_white_edge_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.white_search_scale < 1.0:
        parser.error("--white-search-scale must be >= 1")
    if not 0 <= args.white_canny_low < args.white_canny_high <= 255:
        parser.error("require 0 <= --white-canny-low < --white-canny-high <= 255")
    if args.white_edge_blur_kernel < 3:
        parser.error("--white-edge-blur-kernel must be >= 3")
    if args.white_edge_close_kernel < 1 or args.white_edge_close_iterations < 1:
        parser.error("white edge close parameters must be positive")
    if args.white_edge_min_area <= 0:
        parser.error("--white-edge-min-area must be positive")
    if args.white_mask_close_kernel < 1 or args.white_mask_close_iterations < 1:
        parser.error("white mask close parameters must be positive")
    if args.white_mask_open_kernel < 1:
        parser.error("--white-mask-open-kernel must be positive")
    if not 0.0 <= args.white_contour_epsilon <= 0.05:
        parser.error("--white-contour-epsilon must be in [0,0.05]")
    if args.white_inscribed_angle_range_deg < 0.0:
        parser.error("--white-inscribed-angle-range-deg must be non-negative")
    if args.white_inscribed_angle_step_deg <= 0.0:
        parser.error("--white-inscribed-angle-step-deg must be positive")
    if args.white_inscribed_margin_px < 0:
        parser.error("--white-inscribed-margin-px must be non-negative")
    if args.white_outlier_mad_scale <= 0.0 or args.white_outlier_min_mm <= 0.0:
        parser.error("white outlier thresholds must be positive")
    if args.white_min_inliers < 3 or args.white_min_inliers > args.stable_samples:
        parser.error("--white-min-inliers must be in [3, --stable-samples]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pick a YOLO-detected white workpiece and place it at the red "
            "candidate nearest to the detected white-workpiece center"
        )
    )
    parser.add_argument("--white-model", type=Path, default=DEFAULT_WHITE_MODEL)
    parser.add_argument(
        "--depth-result-topic",
        default="/depth_background_rect_detector/result",
    )
    parser.add_argument("--depth-detection-timeout", type=float, default=30.0)
    parser.add_argument(
        "--red-min-area",
        type=float,
        default=3000.0,
        help="minimum red rectangle area passed to an auto-started detector",
    )
    parser.add_argument(
        "--red-debug-output",
        type=Path,
        default=PROJECT_ROOT / "red_target_debug.png",
        help="saved RGB overlay containing the red rectangle and target points",
    )
    parser.add_argument(
        "--depth-detector-python",
        type=Path,
        default=Path("/usr/bin/python3"),
        help="Python executable used for the auto-started ROS depth detector",
    )
    parser.add_argument(
        "--no-auto-start-depth-detector",
        dest="auto_start_depth_detector",
        action="store_false",
        help="require the depth detector to be started in another terminal",
    )
    parser.set_defaults(auto_start_depth_detector=True)
    parser.add_argument("--host", default="192.168.1.4")
    parser.add_argument("--gripper-port", type=int, default=63352)
    parser.add_argument("--gripper-speed", type=int, default=255)
    parser.add_argument("--gripper-force", type=int, default=255)
    parser.add_argument("--color-topic", default="/camera/color/image_raw")
    parser.add_argument(
        "--depth-topic", default="/camera/aligned_depth_to_color/image_raw"
    )
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument("--handeye-camera-frame", default="camera_link")
    parser.add_argument(
        "--handeye-translation",
        nargs=3,
        type=float,
        default=vision.DEFAULT_HAND_EYE_TRANSLATION,
    )
    parser.add_argument(
        "--handeye-quaternion",
        nargs=4,
        type=float,
        default=vision.DEFAULT_HAND_EYE_QUATERNION,
    )
    parser.add_argument("--tf-timeout", type=float, default=10.0)
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--stable-samples", type=int, default=12)
    parser.add_argument("--association-px", type=float, default=80.0)
    parser.add_argument("--input-timeout", type=float, default=30.0)
    parser.add_argument("--max-sync-delta", type=float, default=0.10)
    parser.add_argument("--depth-radius", type=int, default=5)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--pick-approach-height", type=float, default=0.10)
    parser.add_argument("--pick-descent", type=float, default=0.06)
    parser.add_argument(
        "--post-pick-lift",
        type=float,
        default=0.15,
        help="direct base-Z moveL lift from the closed-gripper grasp pose",
    )
    parser.add_argument(
        "--place-approach-height",
        "--place-drop-height",
        dest="place_approach_height",
        type=float,
        default=0.20,
        help="height above the red slot before the final moveL descent",
    )
    parser.add_argument(
        "--place-descent",
        type=float,
        default=0.08,
        help="base-Z moveL descent from the red slot approach",
    )
    parser.add_argument(
        "--workspace",
        nargs=6,
        type=float,
        default=(-0.45, 0.45, 0.10, 0.55, 0.02, 0.65),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    parser.add_argument("--joint-speed", type=float, default=0.10)
    parser.add_argument("--joint-acc", type=float, default=0.20)
    parser.add_argument("--linear-speed", type=float, default=0.01)
    parser.add_argument("--linear-acc", type=float, default=0.03)
    parser.add_argument("--motion-timeout", type=float, default=60.0)
    parser.add_argument("--gripper-timeout", type=float, default=5.0)
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true")
    add_white_edge_arguments(parser)
    args = parser.parse_args(rospy.myargv()[1:])

    if not args.white_model.is_file():
        parser.error("model does not exist: {}".format(args.white_model))
    if not DEFAULT_DEPTH_DETECTOR.is_file():
        parser.error("depth detector does not exist: {}".format(DEFAULT_DEPTH_DETECTOR))
    if not args.depth_detector_python.is_file():
        parser.error(
            "depth detector Python does not exist: {}".format(
                args.depth_detector_python
            )
        )
    if args.red_min_area < 0.0:
        parser.error("--red-min-area must be non-negative")
    if not 0 <= args.gripper_speed <= 255:
        parser.error("--gripper-speed must be in [0,255]")
    if not 0 <= args.gripper_force <= 255:
        parser.error("--gripper-force must be in [0,255]")
    for name in (
        "pick_approach_height",
        "pick_descent",
        "post_pick_lift",
        "place_approach_height",
        "place_descent",
        "joint_speed",
        "joint_acc",
        "linear_speed",
        "linear_acc",
        "motion_timeout",
        "gripper_timeout",
        "depth_detection_timeout",
    ):
        if getattr(args, name) <= 0.0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.pick_descent >= args.pick_approach_height:
        parser.error("--pick-descent must be less than --pick-approach-height")
    if args.place_descent >= args.place_approach_height:
        parser.error("--place-descent must be less than --place-approach-height")
    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be in (0,1]")
    validate_white_edge_arguments(parser, args)
    return args


def prepare_target_geometry(
    target: Dict[str, Any],
    surface: np.ndarray,
    collector: vision.SensorCollector,
    base_from_camera: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    return obb_on_base_plane(
        target,
        float(surface[2]),
        latest_camera_info(collector),
        base_from_camera,
    )


def robust_white_pick_geometry(
    target: Dict[str, Any],
    pick_fraction: float,
    collector: vision.SensorCollector,
    base_from_camera: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Reject multi-frame 3-D pick outliers, then average retained geometry."""
    samples = list(
        target.get("_captured_samples")
        or target.get("_stable_samples")
        or [target]
    )
    records: List[Dict[str, Any]] = []
    for sample in samples:
        surface = base_surface_point(sample, base_from_camera)
        center, short, long_axis, width, length = prepare_target_geometry(
            sample, surface, collector, base_from_camera
        )
        pick = fractional_point_from_nearer_base_origin(
            center, long_axis, length, pick_fraction
        )
        records.append(
            {
                "center": center,
                "short": short,
                "long": long_axis,
                "width": width,
                "length": length,
                "pick": pick,
            }
        )

    pick_points = np.stack([record["pick"] for record in records])
    median_pick = np.median(pick_points, axis=0)
    residual_mm = np.linalg.norm(pick_points - median_pick, axis=1) * 1000.0
    residual_median = float(np.median(residual_mm))
    mad = float(np.median(np.abs(residual_mm - residual_median)))
    robust_sigma = 1.4826 * mad
    rejection_limit_mm = max(
        float(args.white_outlier_min_mm),
        residual_median + float(args.white_outlier_mad_scale) * robust_sigma,
    )
    inlier_indices = np.flatnonzero(residual_mm <= rejection_limit_mm)
    if inlier_indices.size < args.white_min_inliers:
        raise RuntimeError(
            "white target has only {} inlier frames out of {}; require at "
            "least {} (limit {:.2f} mm, residuals={})".format(
                int(inlier_indices.size),
                len(records),
                args.white_min_inliers,
                rejection_limit_mm,
                np.round(residual_mm, 2).tolist(),
            )
        )

    retained = [records[int(index)] for index in inlier_indices]

    def average_axis(name: str) -> np.ndarray:
        reference = np.asarray(retained[0][name], dtype=np.float64)
        aligned = []
        for record in retained:
            axis = np.asarray(record[name], dtype=np.float64)
            if np.dot(axis, reference) < 0.0:
                axis = -axis
            aligned.append(axis)
        result = np.mean(aligned, axis=0)
        result[2] = 0.0
        norm = float(np.linalg.norm(result))
        if norm < 1e-9:
            raise RuntimeError("averaged white {} axis is degenerate".format(name))
        return result / norm

    result = {
        "center": np.mean([record["center"] for record in retained], axis=0),
        "short": average_axis("short"),
        "long": average_axis("long"),
        "width": float(np.mean([record["width"] for record in retained])),
        "length": float(np.mean([record["length"] for record in retained])),
        "pick": np.mean([record["pick"] for record in retained], axis=0),
        "total_samples": len(records),
        "inlier_samples": int(inlier_indices.size),
        "rejected_samples": int(len(records) - inlier_indices.size),
        "rejection_limit_mm": float(rejection_limit_mm),
        "residuals_mm": residual_mm.tolist(),
    }
    rospy.loginfo(
        "White robust fusion: kept %d/%d frames, rejected=%d, limit=%.2f mm",
        result["inlier_samples"],
        result["total_samples"],
        result["rejected_samples"],
        result["rejection_limit_mm"],
    )
    rospy.loginfo(
        "White pick residuals [mm]: %s",
        np.round(residual_mm, 2).tolist(),
    )
    return result


def base_point_to_color_pixel(
    point_base: Sequence[float],
    camera_info: Any,
    base_from_camera: np.ndarray,
) -> np.ndarray:
    """Project a base_link point into the aligned color image."""
    camera_from_base = np.linalg.inv(base_from_camera)
    point_camera = camera_from_base @ np.append(
        np.asarray(point_base, dtype=np.float64), 1.0
    )
    if point_camera[2] <= 1e-6:
        raise ValueError("red target projects behind the color camera")
    fx, fy = float(camera_info.K[0]), float(camera_info.K[4])
    cx, cy = float(camera_info.K[2]), float(camera_info.K[5])
    return np.array(
        [
            fx * point_camera[0] / point_camera[2] + cx,
            fy * point_camera[1] / point_camera[2] + cy,
        ],
        dtype=np.float64,
    )


def publish_red_debug_image(
    red: Dict[str, Any],
    slot_candidates: Sequence[np.ndarray],
    long_fractions: Sequence[float],
    candidate_white_distances: Sequence[float],
    selected_index: int,
    camera_info: Any,
    base_from_camera: np.ndarray,
    publisher: Any,
    output_path: Path,
) -> None:
    """Draw the depth rectangle and all robot target points on the RGB frame."""
    color_message = red["color_message"]
    debug = vision.color_message_to_bgr(color_message)
    corners = np.rint(np.asarray(red["corners_px"], dtype=np.float64)).astype(
        np.int32
    )
    center = tuple(
        np.rint(np.asarray(red["center_px"], dtype=np.float64)).astype(np.int32)
    )

    # Green: depth-difference rectangle. Blue: its measured center.
    cv2.polylines(debug, [corners], True, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.drawMarker(
        debug, center, (255, 100, 0), cv2.MARKER_CROSS, 24, 3, cv2.LINE_AA
    )
    cv2.putText(
        debug,
        "RED depth rectangle",
        (max(10, int(corners[:, 0].min())), max(30, int(corners[:, 1].min()) - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    candidate_pixels = [
        base_point_to_color_pixel(point, camera_info, base_from_camera)
        for point in slot_candidates
    ]
    if len(candidate_pixels) >= 2:
        endpoints = np.rint(
            np.stack([candidate_pixels[0], candidate_pixels[-1]])
        ).astype(np.int32)
        cv2.line(
            debug,
            tuple(endpoints[0]),
            tuple(endpoints[1]),
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for index, (pixel, fraction) in enumerate(
        zip(candidate_pixels, long_fractions)
    ):
        location = tuple(np.rint(pixel).astype(np.int32))
        selected = index == selected_index
        color = (0, 255, 255) if selected else (255, 255, 0)
        radius = 13 if selected else 8
        thickness = 4 if selected else 2
        cv2.circle(debug, location, radius, color, thickness, cv2.LINE_AA)
        if selected:
            cv2.drawMarker(
                debug,
                location,
                color,
                cv2.MARKER_CROSS,
                32,
                4,
                cv2.LINE_AA,
            )
        distance_m = float(candidate_white_distances[index])
        cv2.putText(
            debug,
            "{}:{:.3f} {:.3f}m{}".format(
                index + 1,
                fraction,
                distance_m,
                " NEAREST WHITE" if selected else "",
            ),
            (location[0] + 12, location[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        debug,
        "green=detected box  cyan=candidates  yellow=robot target",
        (12, debug.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    publisher.publish(vision.bgr_to_image_message(debug, color_message.header))

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), debug):
        rospy.logwarn("Could not save red debug image: %s", output_path)
    else:
        print("red debug image saved:", output_path)


def main() -> None:
    args = parse_args()
    rospy.init_node("visual_pick_and_place")
    collector = vision.SensorCollector(
        args.color_topic, args.depth_topic, args.camera_info_topic
    )
    white_target_pub = rospy.Publisher(
        "~white_robot_target", String, queue_size=1, latch=True
    )
    red_target_pub = rospy.Publisher(
        "~red_robot_target", String, queue_size=1, latch=True
    )
    red_debug_pub = rospy.Publisher(
        "~red_debug_image", Image, queue_size=1, latch=True
    )
    # Constructor needs a model; white is the first stage and is selected
    # randomly by the existing stable-target tracker when multiple are visible.
    args.model = args.white_model
    acquirer = WhiteEdgeTargetAcquirer(args, collector)
    base_from_camera = wait_for_base_camera_matrix(args)
    low, high = vision.parse_bounds(args.workspace)

    white = acquire_with_model(acquirer, args.white_model, "white workpiece")
    white_pick_fraction = 65.0 / 128.0
    white_geometry = robust_white_pick_geometry(
        white,
        white_pick_fraction,
        collector,
        base_from_camera,
        args,
    )
    white_center = white_geometry["center"]
    white_short = white_geometry["short"]
    white_long = white_geometry["long"]
    white_width = white_geometry["width"]
    white_length = white_geometry["length"]
    white_pick_surface = white_geometry["pick"]
    white_angle = base_link_axis_angle_in_ur_base(white_short)

    white_approach_xyz = white_pick_surface.copy()
    white_approach_xyz[2] += args.pick_approach_height
    white_grasp_xyz = white_approach_xyz.copy()
    white_grasp_xyz[2] -= args.pick_descent
    post_pick_lift_xyz = white_grasp_xyz.copy()
    post_pick_lift_xyz[2] += args.post_pick_lift
    validate_workspace(white_approach_xyz, low, high, "white approach")
    validate_workspace(white_grasp_xyz, low, high, "white grasp")
    validate_workspace(post_pick_lift_xyz, low, high, "post-pick lift")

    # In dry-run mode the reference orientation is the nearest known pose. In
    # execute mode this is replaced by the actual current TCP orientation.
    nearby_rotation = GRIPPER_BASE_X_REFERENCE_ROTVEC
    robot = None
    if args.execute:
        robot = UR_BASE(args.host, gripper_port=args.gripper_port)
        connected = robot.connect()
        if not connected.success:
            raise RuntimeError(connected.message)
        current_pose = vision.verify_robot_ready(robot)
        nearby_rotation = np.asarray(current_pose[3:], dtype=np.float64)

    white_rotation = aligned_gripper_rotvec(white_angle, nearby_rotation)
    white_approach_pose = make_ur_pose(white_approach_xyz, white_rotation)
    white_grasp_pose = make_ur_pose(white_grasp_xyz, white_rotation)
    post_pick_lift_pose = make_ur_pose(post_pick_lift_xyz, white_rotation)

    print("\n" + "=" * 76)
    print("WHITE PICK PLAN")
    print("white inscribed-rectangle center base_link:", white_center.round(6).tolist())
    print("white pick point: 65/128 along long edge from endpoint nearer base_link origin")
    print("white pick surface base_link:", white_pick_surface.round(6).tolist())
    print("white inner rectangle short/long [m]: {:.4f} / {:.4f}".format(
        white_width, white_length
    ))
    print(
        "white robust frames: {}/{} kept, {} rejected".format(
            white_geometry["inlier_samples"],
            white_geometry["total_samples"],
            white_geometry["rejected_samples"],
        )
    )
    print("finger line target: white inscribed-rectangle short edge")
    print_pose("white approach UR base", white_approach_pose)
    print_pose("white grasp UR base", white_grasp_pose)
    print_pose("post-pick extra lift UR base", post_pick_lift_pose)
    print("=" * 76)

    white_payload = {
        "frame": "base_link",
        "center_base_link_m": white_center.tolist(),
        "pick_surface_base_link_m": white_pick_surface.tolist(),
        "robust_samples_total": white_geometry["total_samples"],
        "robust_samples_used": white_geometry["inlier_samples"],
        "robust_samples_rejected": white_geometry["rejected_samples"],
        "robust_rejection_limit_mm": white_geometry["rejection_limit_mm"],
        "approach_ur_base_pose": white_approach_pose,
        "grasp_ur_base_pose": white_grasp_pose,
        "post_pick_ur_base_pose": post_pick_lift_pose,
    }
    white_target_pub.publish(String(data=json.dumps(white_payload)))
    rospy.loginfo(
        "Published white robot target: /visual_pick_and_place/white_robot_target"
    )

    # Strict sequence requested by the workflow: only after the white target
    # has been acquired, transformed, printed and published do we invoke the
    # depth-difference detector and acquire the red rectangle.
    ensure_depth_detector(args)
    rospy.loginfo("Waiting for the red depth-difference rectangle...")
    red = acquire_depth_red_target(args, collector)
    red_surface = base_surface_point(red, base_from_camera)
    red_center, red_short, red_long, red_width, red_length = (
        prepare_target_geometry(red, red_surface, collector, base_from_camera)
    )
    long_fractions = (11.0 / 64.0, 25.0 / 64.0, 39.0 / 64.0, 53.0 / 64.0)
    width_fraction = 15/32

    slot_candidates = ordered_fractional_points(
        red_center,
        red_long,
        red_length,
        long_fractions,
    )

    width_offset = (width_fraction - 0.5) * red_width * red_short
    slot_candidates = [
        point + width_offset
        for point in slot_candidates
    ]
    slot_approaches = []
    for index, slot in enumerate(slot_candidates, start=1):
        approach = slot.copy()
        approach[2] += args.place_approach_height
        validate_workspace(approach, low, high, "red slot {} approach".format(index))
        slot_approaches.append(approach)

    # Select the final, width-offset red target closest to the detected white
    # workpiece center.  XY distance is used because placement choice is a
    # top-view decision and the two surfaces may have slightly different Z.
    candidate_white_distances = [
        float(
            np.linalg.norm(
                np.asarray(point, dtype=np.float64)[:2] - white_center[:2]
            )
        )
        for point in slot_candidates
    ]
    selected_index = int(np.argmin(candidate_white_distances))
    selected_slot = slot_candidates[selected_index]
    selected_approach = slot_approaches[selected_index]

    # The line between the two gripper fingers must be parallel to the
    # long edge of the red depth rectangle during placement.
    place_angle = base_link_axis_angle_in_ur_base(red_long)
    place_rotation = aligned_gripper_rotvec(place_angle, white_rotation)
    place_pose = make_ur_pose(selected_approach, place_rotation)
    place_release_xyz = selected_approach.copy()
    place_release_xyz[2] -= args.place_descent
    validate_workspace(place_release_xyz, low, high, "red slot release")
    place_release_pose = make_ur_pose(place_release_xyz, place_rotation)

    publish_red_debug_image(
        red,
        slot_candidates,
        long_fractions,
        candidate_white_distances,
        selected_index,
        latest_camera_info(collector),
        base_from_camera,
        red_debug_pub,
        args.red_debug_output,
    )
    print(
        "red debug ROS topic: /visual_pick_and_place/red_debug_image"
    )

    print("\n" + "=" * 76)
    print("RED DEPTH-DIFFERENCE TARGET / PLACEMENT PLAN")
    print("red depth rectangle center base_link:", red_center.round(6).tolist())
    print("red depth rectangle short/long [m]: {:.4f} / {:.4f}".format(
        red_width, red_length
    ))
    print("white reference center base_link:", white_center.round(6).tolist())
    print("candidate width coordinate: {:.5f}".format(width_fraction))
    for index, (fraction, point, distance_m) in enumerate(
        zip(long_fractions, slot_candidates, candidate_white_distances), start=1
    ):
        print(
            "candidate {} (long={:.3f}, width={:.5f}, white XY distance={:.4f} m) "
            "base_link: {}".format(
                index,
                fraction,
                width_fraction,
                distance_m,
                point.round(6).tolist(),
            )
        )
    print("candidate nearest to white workpiece selected:", selected_index + 1)
    print(
        "selected white XY distance [m]:",
        round(candidate_white_distances[selected_index], 6),
    )
    print("selected target base_link:", selected_slot.round(6).tolist())
    print("finger line target: red rectangle long edge")
    print_pose("red slot approach UR base", place_pose)
    print_pose("red slot release UR base", place_release_pose)
    print(
        "placement mode: approach {:.3f} m above surface, descend {:.3f} m, "
        "release {:.3f} m above surface".format(
            args.place_approach_height,
            args.place_descent,
            args.place_approach_height - args.place_descent,
        )
    )
    print("=" * 76)

    red_payload = {
        "frame": "base_link",
        "rectangle_center_base_link_m": red_center.tolist(),
        "selected_candidate": selected_index + 1,
        "selected_fraction_long": long_fractions[selected_index],
        "selected_fraction_width": width_fraction,
        "selection_strategy": "minimum_white_center_xy_distance",
        "white_reference_center_base_link_m": white_center.tolist(),
        "selected_white_xy_distance_m": candidate_white_distances[selected_index],
        "candidate_white_xy_distances_m": candidate_white_distances,
        "selected_target_base_link_m": selected_slot.tolist(),
        "approach_ur_base_pose": place_pose,
        "release_ur_base_pose": place_release_pose,
    }
    red_target_pub.publish(String(data=json.dumps(red_payload)))
    rospy.loginfo(
        "Published red robot target: /visual_pick_and_place/red_robot_target"
    )

    try:
        # Keep publishing the white-model overlay throughout approach, descent,
        # grasp and retreat, then stop it before depth-based red detection.
        acquirer.start_live_preview(white)
        if args.execute:
            confirm("Type MOVE to execute WHITE PICK: ", args.yes)
            run_motion(
                "open gripper",
                lambda: robot.open_gripper(
                    speed=args.gripper_speed,
                    force=args.gripper_force,
                    timeout_s=args.gripper_timeout,
                ),
            )
            run_motion(
                "moveJ/IK to white approach",
                lambda: robot.move_j_ik(
                    white_approach_pose,
                    args.joint_speed,
                    args.joint_acc,
                    args.motion_timeout,
                ),
            )
            run_motion(
                "moveL to white grasp",
                lambda: robot.move_l(
                    white_grasp_pose,
                    args.linear_speed,
                    args.linear_acc,
                    args.motion_timeout,
                ),
            )
            run_motion(
                "close gripper",
                lambda: robot.close_gripper(
                    speed=args.gripper_speed,
                    force=args.gripper_force,
                    timeout_s=args.gripper_timeout,
                ),
            )
            run_motion(
                "moveL direct post-pick lift",
                lambda: robot.move_l(
                    post_pick_lift_pose,
                    args.linear_speed,
                    args.linear_acc,
                    args.motion_timeout,
                ),
            )

        acquirer.stop_live_preview()

        if not args.execute:
            print("DRY RUN complete: robot and gripper were not moved.")
            print(
                "Depth image: rqt_image_view "
                "/depth_background_rect_detector/debug_image"
            )
            return

        confirm("Type MOVE to execute RED PLACEMENT: ", args.yes)
        run_motion(
            "moveJ/IK to red target nearest to white workpiece",
            lambda: robot.move_j_ik(
                place_pose,
                args.joint_speed,
                args.joint_acc,
                args.motion_timeout,
            ),
        )
        run_motion(
            "moveL descend into red slot",
            lambda: robot.move_l(
                place_release_pose,
                args.linear_speed,
                args.linear_acc,
                args.motion_timeout,
            ),
        )
        run_motion(
            "open gripper and release workpiece",
            lambda: robot.open_gripper(
                speed=args.gripper_speed,
                force=args.gripper_force,
                timeout_s=args.gripper_timeout,
            ),
        )
        print("Full visual pick-and-place sequence completed.")
    finally:
        acquirer.stop_live_preview()
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    main()
