#!/usr/bin/env python3
"""Coordinate two UR3 arms for white-part handoff and red-box placement.

Arm 192.168.1.5 (the historical ``second_hand`` arm) detects and picks a
white rectangle, then carries it to the handoff pose.  Arm 192.168.1.4 grips
the part at that pose, waits ``put_second_hand_time``, and triggers the first
arm to release and return to its observation pose.  After
``wait_second_hand`` from completion of the release, the receiving arm starts
its retract/lift/place sequence while the supplying arm returns concurrently.

No robot or gripper command is sent unless ``--execute`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")

import rospy
import tf2_ros
from sensor_msgs.msg import Image
from std_msgs.msg import String

from complete_process import detect_and_pick as receiver_vision
from complete_process import visual_pick_and_place as red_flow
from complete_process import white_rectangle_pick as supplier_vision
from complete_process.utils.base.ur_base import UR_BASE


DEFAULT_RECEIVER_TARGET_POSE_MM = (
    -187.325487, -290.96447, 339.416991,
    -1.213291, -1.177699, 1.382175,
)
DEFAULT_SUPPLIER_FINAL_POSE = (
    -0.323390523, 0.120168609, 0.364210170,
    -1.489018, -1.057418, 1.446190,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    timing = parser.add_argument_group("handoff timing")
    timing.add_argument(
        "--put_second_hand_time", "--put-second-hand-time",
        dest="put_second_hand_time", type=float, default=0.5,
        help="seconds after receiver grip completion before supplier release",
    )
    timing.add_argument(
        "--wait_second_hand", "--wait-second-hand",
        dest="wait_second_hand", type=float, default=0.5,
        help="seconds after supplier release completion before receiver moves",
    )

    supplier = parser.add_argument_group("white-part supplier (192.168.1.5)")
    supplier.add_argument("--second-hand-host", default="192.168.1.5")
    supplier.add_argument("--second-hand-gripper-port", type=int, default=63352)
    supplier.add_argument("--white-model", type=Path, default=supplier_vision.DEFAULT_MODEL)
    supplier.add_argument("--white-handeye", type=Path, default=supplier_vision.DEFAULT_HANDEYE)
    supplier.add_argument("--white-color-topic", default="/d435/color/image_raw")
    supplier.add_argument("--white-depth-topic", default="/d435/aligned_depth_to_color/image_raw")
    supplier.add_argument("--white-camera-info-topic", default="/d435/color/camera_info")
    supplier.add_argument("--d435-color-offset", type=float, default=0.015)
    supplier.add_argument("--require-camera-tf", action="store_true")
    supplier.add_argument("--confidence", type=float, default=0.25)
    supplier.add_argument("--iou", type=float, default=0.45)
    supplier.add_argument("--imgsz", type=int, default=640)
    supplier.add_argument("--device", default="cpu")
    supplier.add_argument("--max-det", type=int, default=10)
    supplier.add_argument("--capture-frames", type=int, default=12)
    supplier.add_argument("--capture-fps", type=float, default=5.0)
    supplier.add_argument("--stability-px", type=float, default=6.0)
    supplier.add_argument("--white-input-timeout", type=float, default=30.0)
    supplier.add_argument("--white-max-sync-delta", type=float, default=0.10)
    supplier.add_argument("--white-depth-radius", type=int, default=5)
    supplier.add_argument("--white-min-depth", type=float, default=0.10)
    supplier.add_argument("--white-max-depth", type=float, default=1.50)
    supplier.add_argument("--pregrasp-distance", type=float, default=0.10)
    supplier.add_argument("--grasp-distance", type=float, default=0.015)
    supplier.add_argument("--lift-distance", type=float, default=0.13)
    supplier.add_argument("--final-pose", nargs=6, type=float,
                          default=DEFAULT_SUPPLIER_FINAL_POSE)
    supplier.add_argument("--observation-joints", nargs=6, type=float)
    supplier.add_argument("--tcp-offset", nargs=6, type=float,
                          default=(0.0, 0.0, 0.135, 0.0, 0.0, 0.0))
    supplier.add_argument("--movej-speed", type=float, default=0.30)
    supplier.add_argument("--movej-acc", type=float, default=0.30)
    supplier.add_argument("--movel-speed", type=float, default=0.30)
    supplier.add_argument("--movel-acc", type=float, default=0.30)
    supplier.add_argument("--second-hand-motion-timeout", type=float, default=100.0)
    supplier.add_argument("--second-hand-workspace", nargs=6, type=float,
                          default=(-0.45, 0.45, 0.05, 0.75, 0.03, 0.65))

    receiver = parser.add_argument_group("red-box placement receiver (192.168.1.4)")
    receiver.add_argument("--placement-host", default="192.168.1.4")
    receiver.add_argument("--placement-gripper-port", type=int, default=63352)
    receiver.add_argument("--target-pose", nargs=6, type=float,
                          default=DEFAULT_RECEIVER_TARGET_POSE_MM)
    receiver.add_argument("--tool-advance-mm", type=float, default=45.0)
    receiver.add_argument("--post-pick-lift", type=float, default=0.15)
    receiver.add_argument("--slot-index", type=int, choices=(0, 1, 2, 3, 4), default=0)
    receiver.add_argument("--gripper-speed", type=int, default=255)
    receiver.add_argument("--gripper-force", type=int, default=255)
    receiver.add_argument("--gripper-timeout", type=float, default=5.0)
    receiver.add_argument("--depth-result-topic", default="/depth_background_rect_detector/result")
    receiver.add_argument("--depth-detection-timeout", type=float, default=30.0)
    receiver.add_argument("--red-min-area", type=float, default=3000.0)
    receiver.add_argument("--red-debug-output", type=Path,
                          default=PROJECT_ROOT / "output/red_target_debug.png")
    receiver.add_argument("--depth-detector-python", type=Path, default=Path("/usr/bin/python3"))
    receiver.add_argument("--no-auto-start-depth-detector",
                          dest="auto_start_depth_detector", action="store_false")
    receiver.add_argument("--red-color-topic", default="/camera/color/image_raw")
    receiver.add_argument("--red-depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    receiver.add_argument("--red-camera-info-topic", default="/camera/color/camera_info")
    receiver.add_argument("--camera-frame", default="camera_color_optical_frame")
    receiver.add_argument("--handeye-camera-frame", default="camera_link")
    receiver.add_argument("--handeye-translation", nargs=3, type=float,
                          default=receiver_vision.DEFAULT_HAND_EYE_TRANSLATION)
    receiver.add_argument("--handeye-quaternion", nargs=4, type=float,
                          default=receiver_vision.DEFAULT_HAND_EYE_QUATERNION)
    receiver.add_argument("--tf-timeout", type=float, default=10.0)
    receiver.add_argument("--red-input-timeout", type=float, default=30.0)
    receiver.add_argument("--red-max-sync-delta", type=float, default=0.10)
    receiver.add_argument("--red-depth-radius", type=int, default=5)
    receiver.add_argument("--red-min-depth", type=float, default=0.15)
    receiver.add_argument("--red-max-depth", type=float, default=2.0)
    receiver.add_argument("--place-approach-height", type=float, default=0.20)
    receiver.add_argument("--place-descent", type=float, default=0.06)
    receiver.add_argument("--workspace", nargs=6, type=float,
                          default=(-0.45, 0.45, 0.10, 0.55, 0.02, 0.65))
    receiver.add_argument("--joint-speed", type=float, default=0.30)
    receiver.add_argument("--joint-acc", type=float, default=0.30)
    receiver.add_argument("--linear-speed", type=float, default=0.30)
    receiver.add_argument("--linear-acc", type=float, default=0.30)
    receiver.add_argument("--motion-timeout", type=float, default=60.0)

    execution = parser.add_argument_group("execution")
    execution.add_argument("--execute", action="store_true")
    execution.add_argument("--full-auto", action="store_true",
                           help="skip the single final safety confirmation")
    parser.set_defaults(auto_start_depth_detector=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO white-part pick, timed dual-arm handoff, red-box placement"
    )
    add_arguments(parser)
    args = parser.parse_args(rospy.myargv()[1:])
    if not args.white_model.is_file():
        parser.error("white model does not exist: {}".format(args.white_model))
    if not args.white_handeye.is_file():
        parser.error("white hand-eye calibration does not exist: {}".format(args.white_handeye))
    if not red_flow.DEFAULT_DEPTH_DETECTOR.is_file():
        parser.error("red depth detector does not exist: {}".format(red_flow.DEFAULT_DEPTH_DETECTOR))
    if not args.depth_detector_python.is_file():
        parser.error("depth detector Python does not exist: {}".format(args.depth_detector_python))
    finite_groups = (args.target_pose, args.final_pose, args.tcp_offset,
                     args.workspace, args.second_hand_workspace)
    if any(not np.all(np.isfinite(np.asarray(values, dtype=np.float64)))
           for values in finite_groups):
        parser.error("pose and workspace values must be finite")
    if args.second_hand_host == args.placement_host:
        parser.error("the two robot hosts must be different")
    if not 0.0 < args.confidence <= 1.0 or not 0.0 < args.iou <= 1.0:
        parser.error("--confidence and --iou must be in (0,1]")
    if not 0.0 < args.grasp_distance < args.pregrasp_distance:
        parser.error("require 0 < --grasp-distance < --pregrasp-distance")
    if args.place_descent >= args.place_approach_height:
        parser.error("--place-descent must be less than --place-approach-height")
    if args.tool_advance_mm <= 0.0 or args.post_pick_lift < 0.0:
        parser.error("tool advance must be positive and post-pick lift non-negative")
    for name in (
        "put_second_hand_time", "wait_second_hand", "capture_fps",
        "white_input_timeout", "white_max_sync_delta", "lift_distance",
        "movej_speed", "movej_acc", "movel_speed", "movel_acc",
        "second_hand_motion_timeout", "depth_detection_timeout",
        "red_input_timeout", "red_max_sync_delta", "place_approach_height",
        "place_descent", "joint_speed", "joint_acc", "linear_speed",
        "linear_acc", "motion_timeout", "gripper_timeout",
    ):
        value = getattr(args, name)
        minimum_ok = value >= 0.0 if name in (
            "put_second_hand_time", "wait_second_hand", "capture_fps"
        ) else value > 0.0
        if not math.isfinite(value) or not minimum_ok:
            parser.error("invalid --{}".format(name.replace("_", "-")))
    if args.capture_frames <= 0 or args.imgsz <= 0 or args.max_det <= 0:
        parser.error("capture frames, image size and max detections must be positive")
    for values, label in ((args.workspace, "workspace"),
                          (args.second_hand_workspace, "second-hand-workspace")):
        try:
            receiver_vision.parse_bounds(values)
        except ValueError as error:
            parser.error("invalid --{}: {}".format(label, error))
    return args


def require(name: str, result: Any) -> None:
    print("{}: {}".format(name, result))
    if not result.success:
        raise RuntimeError("{} failed: {}".format(name, result.message))


def supplier_acquirer_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.white_model, confidence=args.confidence, iou=args.iou,
        imgsz=args.imgsz, device=args.device, max_det=args.max_det,
        capture_frames=args.capture_frames, capture_fps=args.capture_fps,
        stability_px=args.stability_px, input_timeout=args.white_input_timeout,
        max_sync_delta=args.white_max_sync_delta,
        depth_radius=args.white_depth_radius,
        min_depth=args.white_min_depth, max_depth=args.white_max_depth,
    )


def supplier_camera_matrix(args: argparse.Namespace, actual_pose: np.ndarray,
                           active_tcp_offset: np.ndarray) -> np.ndarray:
    tool_from_d435 = supplier_vision.load_tool_from_d435(args.white_handeye)
    buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    listener = tf2_ros.TransformListener(buffer)  # noqa: F841
    try:
        internal = buffer.lookup_transform(
            "d435_link", "d435_color_optical_frame", rospy.Time(0), rospy.Duration(3.0)
        )
        d435_from_optical = supplier_vision.transform_message_to_matrix(internal.transform)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException) as error:
        if args.require_camera_tf:
            raise RuntimeError("D435 internal optical TF unavailable: {}".format(error))
        rospy.logwarn("Using nominal D435 optical transform: %s", error)
        d435_from_optical = supplier_vision.nominal_d435_from_color_optical(
            args.d435_color_offset
        )
    base_from_active_tcp = supplier_vision.pose_to_matrix(actual_pose)
    tool_from_active_tcp = supplier_vision.pose_to_matrix(active_tcp_offset)
    return base_from_active_tcp @ np.linalg.inv(tool_from_active_tcp) @ tool_from_d435 @ d435_from_optical


def plan_supplier(args: argparse.Namespace, robot: UR_BASE,
                  collector: supplier_vision.SensorCollector) -> Dict[str, Any]:
    actual = np.asarray(robot.get_tcp_pose_or_raise(), dtype=np.float64)
    active_tcp = np.asarray(args.tcp_offset, dtype=np.float64)
    if robot.rtde_c is not None:
        active_tcp = np.asarray(robot.rtde_c.getTCPOffset(), dtype=np.float64)
    base_from_optical = supplier_camera_matrix(args, actual, active_tcp)
    target = supplier_vision.TargetAcquirer(
        supplier_acquirer_args(args), collector
    ).acquire()
    target_base = (base_from_optical @ np.append(target["point_camera"], 1.0))[:3]
    camera_origin = base_from_optical[:3, 3]
    camera_axis = supplier_vision.normalize(target_base - camera_origin, "camera-to-target axis")
    rotation, finger_axis, approach_axis = supplier_vision.desired_tcp_rotation(
        camera_axis, supplier_vision.pose_to_matrix(actual)[:3, :3]
    )
    pregrasp_xyz = target_base - approach_axis * args.pregrasp_distance
    grasp_xyz = target_base - approach_axis * args.grasp_distance
    rotvec = supplier_vision.matrix_to_rotvec(rotation)
    pregrasp = pregrasp_xyz.tolist() + rotvec.tolist()
    grasp = grasp_xyz.tolist() + rotvec.tolist()
    lift = np.asarray(grasp, dtype=np.float64)
    lift[2] += args.lift_distance
    final = np.asarray(args.final_pose, dtype=np.float64)
    low, high = receiver_vision.parse_bounds(args.second_hand_workspace)
    for point, label in ((pregrasp_xyz, "supplier pregrasp"),
                         (grasp_xyz, "supplier grasp"),
                         (lift[:3], "supplier lift"),
                         (final[:3], "supplier handoff")):
        supplier_vision.validate_workspace(point, low, high, label)
    return {
        "target": target, "pregrasp": pregrasp, "grasp": grasp,
        "lift": lift.tolist(), "handoff": final.tolist(),
        "finger_axis": finger_axis.tolist(), "approach_axis": approach_axis.tolist(),
    }


def receiver_flow_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        depth_result_topic=args.depth_result_topic,
        depth_detection_timeout=args.depth_detection_timeout,
        red_min_area=args.red_min_area, red_debug_output=args.red_debug_output,
        depth_detector_python=args.depth_detector_python,
        auto_start_depth_detector=args.auto_start_depth_detector,
        color_topic=args.red_color_topic, depth_topic=args.red_depth_topic,
        camera_info_topic=args.red_camera_info_topic,
        camera_frame=args.camera_frame,
        handeye_camera_frame=args.handeye_camera_frame,
        handeye_translation=args.handeye_translation,
        handeye_quaternion=args.handeye_quaternion,
        tf_timeout=args.tf_timeout, input_timeout=args.red_input_timeout,
        max_sync_delta=args.red_max_sync_delta,
        depth_radius=args.red_depth_radius, min_depth=args.red_min_depth,
        max_depth=args.red_max_depth,
    )


def ur_pose_from_mm(values: Sequence[float]) -> List[float]:
    pose = np.asarray(values, dtype=np.float64).copy()
    pose[:3] *= 0.001
    return pose.tolist()


def tool_z_pose(pose: Sequence[float], distance_m: float) -> List[float]:
    result = np.asarray(pose, dtype=np.float64).copy()
    result[:3] += red_flow.rotvec_to_matrix(result[3:])[:, 2] * distance_m
    return result.tolist()


def plan_receiver(args: argparse.Namespace, collector: receiver_vision.SensorCollector,
                  debug_pub: Any) -> Dict[str, Any]:
    flow_args = receiver_flow_args(args)
    base_from_camera = red_flow.wait_for_base_camera_matrix(flow_args)
    low, high = receiver_vision.parse_bounds(args.workspace)
    approach = ur_pose_from_mm(args.target_pose)
    grasp = tool_z_pose(approach, args.tool_advance_mm * 0.001)
    approach_bl = receiver_vision.base_link_point_to_ur_base(approach[:3])
    grasp_bl = receiver_vision.base_link_point_to_ur_base(grasp[:3])
    lifted_bl = approach_bl.copy()
    lifted_bl[2] += args.post_pick_lift
    lifted = receiver_vision.base_link_point_to_ur_base(lifted_bl).tolist() + approach[3:]
    for point, label in ((approach_bl, "receiver handoff approach"),
                         (grasp_bl, "receiver handoff grasp"),
                         (lifted_bl, "receiver post-pick lift")):
        red_flow.validate_workspace(point, low, high, label)

    red_flow.ensure_depth_detector(flow_args)
    red = red_flow.acquire_depth_red_target(flow_args, collector)
    surface = red_flow.base_surface_point(red, base_from_camera)
    center, short, long_axis, width, length = red_flow.prepare_target_geometry(
        red, surface, collector, base_from_camera
    )
    fractions = (11.0 / 64.0, 25.0 / 64.0, 39.0 / 64.0, 105.0 / 128.0)
    candidates = red_flow.ordered_fractional_points(center, long_axis, length, fractions)
    candidates = [point + (35.0 / 64.0 - 0.5) * width * short for point in candidates]
    distances = [float(np.linalg.norm(point[:2] - approach_bl[:2])) for point in candidates]
    selected_index = args.slot_index - 1 if args.slot_index else int(np.argmin(distances))
    selected = candidates[selected_index]
    place_xyz = selected.copy()
    place_xyz[2] += args.place_approach_height
    release_xyz = place_xyz.copy()
    release_xyz[2] -= args.place_descent
    red_flow.validate_workspace(place_xyz, low, high, "red slot approach")
    red_flow.validate_workspace(release_xyz, low, high, "red slot release")
    angle = red_flow.base_link_axis_angle_in_ur_base(long_axis)
    rotation = red_flow.aligned_gripper_rotvec(angle, approach[3:])
    place = red_flow.make_ur_pose(place_xyz, rotation)
    release = red_flow.make_ur_pose(release_xyz, rotation)
    red_flow.publish_red_debug_image(
        red, candidates, fractions, distances, selected_index,
        red_flow.latest_camera_info(collector), base_from_camera,
        debug_pub, args.red_debug_output,
    )
    return {
        "approach": approach, "grasp": grasp, "lifted": lifted,
        "place": place, "release": release, "slot": selected_index + 1,
        "red_center": center.tolist(), "candidates": [p.tolist() for p in candidates],
        "distances": distances,
    }


def print_plan(supplier: Dict[str, Any], receiver: Dict[str, Any],
               args: argparse.Namespace) -> None:
    payload = {
        "supplier_pregrasp": supplier["pregrasp"],
        "supplier_grasp": supplier["grasp"],
        "handoff_pose": supplier["handoff"],
        "receiver_handoff_approach": receiver["approach"],
        "receiver_handoff_grasp": receiver["grasp"],
        "selected_red_slot": receiver["slot"],
        "red_place_approach": receiver["place"],
        "red_release": receiver["release"],
        "put_second_hand_time": args.put_second_hand_time,
        "wait_second_hand": args.wait_second_hand,
    }
    print("\n" + "=" * 78)
    print("DUAL-ARM PLAN")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("white overlay: /two_hand_pick_and_place/debug_image")
    print("red live view: /depth_background_rect_detector/debug_image")
    print("red target overlay: /two_hand_pick_and_place/red_debug_image")
    print("=" * 78)


def move_supplier_to_observation(args: argparse.Namespace, robot: UR_BASE) -> None:
    if args.observation_joints is not None:
        require("second_hand moveJ observation joints", robot.move_j(
            args.observation_joints, args.movej_speed, args.movej_acc,
            args.second_hand_motion_timeout,
        ))
    else:
        require("second_hand moveJ/IK observation pose", robot.move_j_ik(
            supplier_vision.OBSERVATION_POSE.tolist(), args.movej_speed,
            args.movej_acc, args.second_hand_motion_timeout,
        ))


def execute_sequence(args: argparse.Namespace, supplier_robot: UR_BASE,
                     receiver_robot: UR_BASE, supplier: Dict[str, Any],
                     receiver: Dict[str, Any]) -> None:
    require("second_hand moveJ pregrasp", supplier_robot.move_j_ik(
        supplier["pregrasp"], args.movej_speed, args.movej_acc,
        args.second_hand_motion_timeout,
    ))
    require("second_hand moveL grasp", supplier_robot.move_l(
        supplier["grasp"], args.movel_speed, args.movel_acc,
        args.second_hand_motion_timeout,
    ))
    require("second_hand close gripper", supplier_robot.close_gripper(
        speed=255, force=255, timeout_s=args.gripper_timeout,
    ))
    actual_grasp = np.asarray(supplier_robot.get_tcp_pose_or_raise(), dtype=np.float64)
    actual_lift = actual_grasp.copy()
    actual_lift[2] += args.lift_distance
    low, high = receiver_vision.parse_bounds(args.second_hand_workspace)
    supplier_vision.validate_workspace(actual_lift[:3], low, high, "actual supplier lift")
    require("second_hand lift", supplier_robot.move_l(
        actual_lift.tolist(), args.movel_speed, args.movel_acc,
        args.second_hand_motion_timeout,
    ))
    require("second_hand move to handoff pose", supplier_robot.move_j_ik(
        supplier["handoff"], args.movej_speed, args.movej_acc,
        args.second_hand_motion_timeout,
    ))

    require("placement arm open gripper", receiver_robot.open_gripper(
        speed=args.gripper_speed, force=args.gripper_force,
        timeout_s=args.gripper_timeout,
    ))
    require("placement arm moveJ to handoff approach", receiver_robot.move_j_ik(
        receiver["approach"], args.joint_speed, args.joint_acc,
        args.motion_timeout,
    ))
    require("placement arm moveL to handoff grasp", receiver_robot.move_l(
        receiver["grasp"], args.linear_speed, args.linear_acc,
        args.motion_timeout,
    ))
    require("placement arm close gripper", receiver_robot.close_gripper(
        speed=args.gripper_speed, force=args.gripper_force,
        timeout_s=args.gripper_timeout,
    ))

    print("Receiver grip completed; waiting {:.3f}s before supplier release".format(
        args.put_second_hand_time
    ))
    time.sleep(args.put_second_hand_time)
    require("second_hand release handoff", supplier_robot.open_gripper(
        speed=255, force=255, timeout_s=args.gripper_timeout,
    ))
    released_at = time.monotonic()

    return_errors: List[BaseException] = []
    def return_supplier() -> None:
        try:
            move_supplier_to_observation(args, supplier_robot)
        except BaseException as error:  # stored and raised on the main thread
            return_errors.append(error)

    return_thread = threading.Thread(
        target=return_supplier, name="second-hand-return", daemon=False
    )
    return_thread.start()
    try:
        remaining = args.wait_second_hand - (time.monotonic() - released_at)
        if remaining > 0.0:
            print("Waiting {:.3f}s from supplier release before receiver motion".format(remaining))
            time.sleep(remaining)
        if return_errors:
            raise RuntimeError("second_hand return failed: {}".format(return_errors[0]))
        require("placement arm retract from handoff", receiver_robot.move_l(
            receiver["approach"], args.linear_speed, args.linear_acc,
            args.motion_timeout,
        ))
        if args.post_pick_lift > 0.0:
            require("placement arm post-pick lift", receiver_robot.move_l(
                receiver["lifted"], args.linear_speed, args.linear_acc,
                args.motion_timeout,
            ))
        require("placement arm moveJ to red slot {}".format(receiver["slot"]),
                receiver_robot.move_j_ik(
                    receiver["place"], args.joint_speed, args.joint_acc,
                    args.motion_timeout,
                ))
        require("placement arm descend into red slot", receiver_robot.move_l(
            receiver["release"], args.linear_speed, args.linear_acc,
            args.motion_timeout,
        ))
        require("placement arm release workpiece", receiver_robot.open_gripper(
            speed=args.gripper_speed, force=args.gripper_force,
            timeout_s=args.gripper_timeout,
        ))
    finally:
        return_thread.join()
    if return_errors:
        raise RuntimeError("second_hand return failed: {}".format(return_errors[0]))


def main() -> None:
    args = parse_args()
    rospy.init_node("two_hand_pick_and_place")
    white_collector = supplier_vision.SensorCollector(
        args.white_color_topic, args.white_depth_topic,
        args.white_camera_info_topic,
    )
    red_collector = receiver_vision.SensorCollector(
        args.red_color_topic, args.red_depth_topic, args.red_camera_info_topic,
    )
    red_pub = rospy.Publisher("~red_debug_image", Image, queue_size=1, latch=True)
    plan_pub = rospy.Publisher("~plan", String, queue_size=1, latch=True)

    supplier_robot = UR_BASE(
        args.second_hand_host,
        gripper_port=args.second_hand_gripper_port if args.execute else None,
        connect_control=args.execute,
    )
    receiver_robot = UR_BASE(
        args.placement_host,
        gripper_port=args.placement_gripper_port if args.execute else None,
        connect_control=args.execute,
    )
    try:
        require("connect second_hand arm", supplier_robot.connect())
        supplier_vision.verify_robot_ready(supplier_robot)
        if args.execute:
            require("connect placement arm", receiver_robot.connect())
            receiver_vision.verify_robot_ready(receiver_robot)
            move_supplier_to_observation(args, supplier_robot)
            require("second_hand open gripper", supplier_robot.open_gripper(
                speed=255, force=255, timeout_s=args.gripper_timeout,
            ))
            time.sleep(1.0)
        else:
            print("DRY RUN: supplier uses its current TCP pose; neither arm will move.")

        supplier_plan = plan_supplier(args, supplier_robot, white_collector)
        receiver_plan = plan_receiver(args, red_collector, red_pub)
        print_plan(supplier_plan, receiver_plan, args)
        plan_pub.publish(String(data=json.dumps({
            "supplier_pregrasp": supplier_plan["pregrasp"],
            "supplier_grasp": supplier_plan["grasp"],
            "supplier_lift": supplier_plan["lift"],
            "handoff_pose": supplier_plan["handoff"],
            "receiver_handoff_approach": receiver_plan["approach"],
            "receiver_handoff_grasp": receiver_plan["grasp"],
            "selected_red_slot": receiver_plan["slot"],
            "red_candidates": receiver_plan["candidates"],
            "red_place_approach": receiver_plan["place"],
            "red_release": receiver_plan["release"],
            "put_second_hand_time": args.put_second_hand_time,
            "wait_second_hand": args.wait_second_hand,
        })))
        if not args.execute:
            print("DRY RUN complete: no robot/gripper command was sent.")
            return
        if not args.full_auto:
            answer = input(
                "确认双臂路径安全且工件/红盒就位，输入 MOVE 开始连续协同流程："
            ).strip()
            if answer != "MOVE":
                raise RuntimeError("operator cancelled dual-arm execution")
        execute_sequence(args, supplier_robot, receiver_robot,
                         supplier_plan, receiver_plan)
        print("Dual-arm pick, handoff and placement completed.")
    finally:
        receiver_robot.disconnect()
        supplier_robot.disconnect()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by operator.")
        raise SystemExit(130)
    except (RuntimeError, ValueError, FileNotFoundError, TimeoutError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
