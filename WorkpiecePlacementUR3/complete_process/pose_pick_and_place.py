#!/usr/bin/env python3
"""Pick from a fixed UR TCP pose and place into a depth-detected red slot.

The fixed pose is supplied in millimetres/radians in UR controller Base
coordinates.  The grasp pose is obtained by translating the TCP along its
local +Z axis.  Red-board detection and placement geometry are shared with
visual_pick_and_place.py.

Robot and gripper motion is disabled unless --execute is supplied.  In
execute mode every individual action requires an Enter-key confirmation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from complete_process import detect_and_pick as vision
from complete_process import visual_pick_and_place as visual_flow
from complete_process.utils.base.ur_base import UR_BASE


DEFAULT_TARGET_POSE_MM = (
    -217.178216,
    -290.346438,
    383.249126,
    -1.010856,
    -1.374927,
    1.180301,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pick at a fixed UR TCP pose, advance along tool +Z, then place "
            "into a depth-detected red-board slot"
        )
    )
    parser.add_argument(
        "--target-pose",
        nargs=6,
        type=float,
        default=DEFAULT_TARGET_POSE_MM,
        metavar=("X_MM", "Y_MM", "Z_MM", "RX", "RY", "RZ"),
        help=(
            "fixed pre-grasp TCP pose in UR Base coordinates; XYZ is in mm "
            "and the rotation vector is in rad"
        ),
    )
    parser.add_argument(
        "--tool-advance-mm",
        type=float,
        default=32.0,
        help="moveL distance along TCP local +Z before grasping (default: 32 mm)",
    )
    parser.add_argument(
        "--post-pick-lift",
        type=float,
        default=0.15,
        help="base-Z lift in metres after retracting to the fixed pose",
    )
    parser.add_argument(
        "--slot-index",
        type=int,
        choices=(0, 1, 2, 3, 4),
        default=0,
        help=(
            "red slot to use (1..4); 0 selects the slot whose XY position is "
            "nearest to the fixed pick pose"
        ),
    )

    parser.add_argument("--host", default="192.168.1.4")
    parser.add_argument("--gripper-port", type=int, default=63352)
    parser.add_argument("--gripper-speed", type=int, default=255)
    parser.add_argument("--gripper-force", type=int, default=255)

    parser.add_argument(
        "--depth-result-topic",
        default="/depth_background_rect_detector/result",
    )
    parser.add_argument("--depth-detection-timeout", type=float, default=30.0)
    parser.add_argument("--red-min-area", type=float, default=3000.0)
    parser.add_argument(
        "--red-debug-output",
        type=Path,
        default=visual_flow.PROJECT_ROOT / "pose_red_target_debug.png",
    )
    parser.add_argument(
        "--depth-detector-python",
        type=Path,
        default=Path("/usr/bin/python3"),
    )
    parser.add_argument(
        "--no-auto-start-depth-detector",
        dest="auto_start_depth_detector",
        action="store_false",
        help="use a depth detector already running in another terminal",
    )
    parser.set_defaults(auto_start_depth_detector=True)

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
    parser.add_argument("--input-timeout", type=float, default=30.0)
    parser.add_argument("--max-sync-delta", type=float, default=0.10)
    parser.add_argument("--depth-radius", type=int, default=5)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=2.0)

    parser.add_argument("--place-approach-height", type=float, default=0.20)
    parser.add_argument("--place-descent", type=float, default=0.08)
    parser.add_argument(
        "--workspace",
        nargs=6,
        type=float,
        default=(-0.45, 0.45, 0.10, 0.55, 0.02, 0.65),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="allowed base_link workspace in metres",
    )
    parser.add_argument("--joint-speed", type=float, default=0.10)
    parser.add_argument("--joint-acc", type=float, default=0.20)
    parser.add_argument("--linear-speed", type=float, default=0.01)
    parser.add_argument("--linear-acc", type=float, default=0.03)
    parser.add_argument("--motion-timeout", type=float, default=60.0)
    parser.add_argument("--gripper-timeout", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true")

    args = parser.parse_args(rospy.myargv()[1:])
    if not visual_flow.DEFAULT_DEPTH_DETECTOR.is_file():
        parser.error(
            "depth detector does not exist: {}".format(
                visual_flow.DEFAULT_DEPTH_DETECTOR
            )
        )
    if not args.depth_detector_python.is_file():
        parser.error(
            "depth detector Python does not exist: {}".format(
                args.depth_detector_python
            )
        )
    if not np.all(np.isfinite(np.asarray(args.target_pose, dtype=np.float64))):
        parser.error("--target-pose must contain six finite values")
    if not math.isfinite(args.tool_advance_mm) or args.tool_advance_mm <= 0.0:
        parser.error("--tool-advance-mm must be positive")
    if not math.isfinite(args.post_pick_lift) or args.post_pick_lift < 0.0:
        parser.error("--post-pick-lift must be non-negative")
    if args.red_min_area < 0.0:
        parser.error("--red-min-area must be non-negative")
    if not 0 <= args.gripper_speed <= 255:
        parser.error("--gripper-speed must be in [0,255]")
    if not 0 <= args.gripper_force <= 255:
        parser.error("--gripper-force must be in [0,255]")
    for name in (
        "depth_detection_timeout",
        "input_timeout",
        "max_sync_delta",
        "min_depth",
        "max_depth",
        "place_approach_height",
        "place_descent",
        "joint_speed",
        "joint_acc",
        "linear_speed",
        "linear_acc",
        "motion_timeout",
        "gripper_timeout",
    ):
        if getattr(args, name) <= 0.0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.place_descent >= args.place_approach_height:
        parser.error("--place-descent must be less than --place-approach-height")
    if args.depth_radius < 0:
        parser.error("--depth-radius must be non-negative")
    return args


def ur_pose_from_mm(values: Sequence[float]) -> List[float]:
    pose = np.asarray(values, dtype=np.float64).copy()
    pose[:3] *= 0.001
    return pose.tolist()


def translate_along_tool_z(pose: Sequence[float], distance_m: float) -> List[float]:
    result = np.asarray(pose, dtype=np.float64).copy()
    rotation = visual_flow.rotvec_to_matrix(result[3:])
    result[:3] += rotation[:, 2] * float(distance_m)
    return result.tolist()


def ur_base_point_to_base_link(point: Sequence[float]) -> np.ndarray:
    """The base/base_link conversion is its own inverse (Rz(pi))."""
    return vision.base_link_point_to_ur_base(point)


def confirm_and_run(name: str, action: Callable[[], Any]) -> None:
    answer = input(
        "\n下一动作：{}\n确认周围安全后按回车执行；输入 q 后回车取消：".format(name)
    ).strip().lower()
    if answer:
        if answer == "q":
            raise RuntimeError("operator cancelled before {}".format(name))
        raise RuntimeError("未收到空回车确认，已取消 {}".format(name))
    visual_flow.run_motion(name, action)


def print_pose_mm(name: str, pose: Sequence[float]) -> None:
    values = np.asarray(pose, dtype=np.float64).copy()
    values[:3] *= 1000.0
    print("{} [mm, rad]: {}".format(name, values.round(6).tolist()))


def main() -> None:
    args = parse_args()
    rospy.init_node("pose_pick_and_place")
    collector = vision.SensorCollector(
        args.color_topic, args.depth_topic, args.camera_info_topic
    )
    red_target_pub = rospy.Publisher(
        "~red_robot_target", String, queue_size=1, latch=True
    )
    red_debug_pub = rospy.Publisher(
        "~red_debug_image", Image, queue_size=1, latch=True
    )

    base_from_camera = visual_flow.wait_for_base_camera_matrix(args)
    low, high = vision.parse_bounds(args.workspace)

    pick_approach_pose = ur_pose_from_mm(args.target_pose)
    pick_grasp_pose = translate_along_tool_z(
        pick_approach_pose, args.tool_advance_mm * 0.001
    )
    tool_delta = (
        np.asarray(pick_grasp_pose[:3]) - np.asarray(pick_approach_pose[:3])
    )
    pick_approach_base_link = ur_base_point_to_base_link(pick_approach_pose[:3])
    pick_grasp_base_link = ur_base_point_to_base_link(pick_grasp_pose[:3])
    post_pick_lift_base_link = pick_approach_base_link.copy()
    post_pick_lift_base_link[2] += args.post_pick_lift
    post_pick_lift_pose = (
        vision.base_link_point_to_ur_base(post_pick_lift_base_link).tolist()
        + pick_approach_pose[3:]
    )
    visual_flow.validate_workspace(
        pick_approach_base_link, low, high, "fixed pick approach"
    )
    visual_flow.validate_workspace(pick_grasp_base_link, low, high, "fixed grasp")
    visual_flow.validate_workspace(
        post_pick_lift_base_link, low, high, "post-pick lift"
    )

    visual_flow.ensure_depth_detector(args)
    rospy.loginfo("Waiting for the red depth-difference rectangle...")
    red = visual_flow.acquire_depth_red_target(args, collector)
    red_surface = visual_flow.base_surface_point(red, base_from_camera)
    red_center, red_short, red_long, red_width, red_length = (
        visual_flow.prepare_target_geometry(
            red, red_surface, collector, base_from_camera
        )
    )

    long_fractions = (11.0 / 64.0, 25.0 / 64.0, 39.0 / 64.0, 53.0 / 64.0)
    width_fraction = 15.0 / 32.0
    slot_candidates = visual_flow.ordered_fractional_points(
        red_center, red_long, red_length, long_fractions
    )
    width_offset = (width_fraction - 0.5) * red_width * red_short
    slot_candidates = [point + width_offset for point in slot_candidates]
    candidate_pick_distances = [
        float(np.linalg.norm(point[:2] - pick_approach_base_link[:2]))
        for point in slot_candidates
    ]
    selected_index = (
        args.slot_index - 1
        if args.slot_index
        else int(np.argmin(candidate_pick_distances))
    )
    selected_slot = slot_candidates[selected_index]

    selected_approach = selected_slot.copy()
    selected_approach[2] += args.place_approach_height
    release_xyz = selected_approach.copy()
    release_xyz[2] -= args.place_descent
    visual_flow.validate_workspace(
        selected_approach, low, high, "red slot approach"
    )
    visual_flow.validate_workspace(release_xyz, low, high, "red slot release")

    place_angle = visual_flow.base_link_axis_angle_in_ur_base(red_long)
    place_rotation = visual_flow.aligned_gripper_rotvec(
        place_angle, pick_approach_pose[3:]
    )
    place_pose = visual_flow.make_ur_pose(selected_approach, place_rotation)
    place_release_pose = visual_flow.make_ur_pose(release_xyz, place_rotation)

    visual_flow.publish_red_debug_image(
        red,
        slot_candidates,
        long_fractions,
        candidate_pick_distances,
        selected_index,
        visual_flow.latest_camera_info(collector),
        base_from_camera,
        red_debug_pub,
        args.red_debug_output,
    )

    print("\n" + "=" * 76)
    print("FIXED-POSE PICK PLAN")
    print_pose_mm("fixed approach pose", pick_approach_pose)
    print("tool +Z advance [mm]:", round(args.tool_advance_mm, 6))
    print("tool advance delta in UR Base [mm]:", (tool_delta * 1000.0).round(6).tolist())
    print_pose_mm("grasp pose", pick_grasp_pose)
    print_pose_mm("retracted and lifted pose", post_pick_lift_pose)
    print("\nRED DEPTH-DIFFERENCE PLACEMENT PLAN")
    print("red rectangle center base_link [m]:", red_center.round(6).tolist())
    print("red rectangle short/long [m]: {:.4f} / {:.4f}".format(
        red_width, red_length
    ))
    for index, (fraction, point, distance_m) in enumerate(
        zip(long_fractions, slot_candidates, candidate_pick_distances), start=1
    ):
        print(
            "slot {} (long={:.6f}, width={:.6f}, pick XY distance={:.4f} m) "
            "base_link: {}".format(
                index, fraction, width_fraction, distance_m, point.round(6).tolist()
            )
        )
    strategy = "explicit_slot_index" if args.slot_index else "nearest_fixed_pick_xy"
    print("selected slot:", selected_index + 1, "strategy:", strategy)
    visual_flow.print_pose("red slot approach UR Base", place_pose)
    visual_flow.print_pose("red slot release UR Base", place_release_pose)
    print("red detector live image: /depth_background_rect_detector/debug_image")
    print("latched placement overlay: /pose_pick_and_place/red_debug_image")
    print("=" * 76)

    payload = {
        "fixed_pick_pose_ur_base_m_rad": pick_approach_pose,
        "tool_advance_m": args.tool_advance_mm * 0.001,
        "grasp_pose_ur_base_m_rad": pick_grasp_pose,
        "rectangle_center_base_link_m": red_center.tolist(),
        "selected_candidate": selected_index + 1,
        "selection_strategy": strategy,
        "candidate_pick_xy_distances_m": candidate_pick_distances,
        "selected_target_base_link_m": selected_slot.tolist(),
        "approach_ur_base_pose": place_pose,
        "release_ur_base_pose": place_release_pose,
    }
    red_target_pub.publish(String(data=json.dumps(payload)))

    if not args.execute:
        print("DRY RUN complete: robot and gripper were not connected or moved.")
        return

    robot = UR_BASE(args.host, gripper_port=args.gripper_port)
    try:
        connected = robot.connect()
        if not connected.success:
            raise RuntimeError(connected.message)
        vision.verify_robot_ready(robot)

        confirm_and_run(
            "打开夹爪",
            lambda: robot.open_gripper(
                speed=args.gripper_speed,
                force=args.gripper_force,
                timeout_s=args.gripper_timeout,
            ),
        )
        confirm_and_run(
            "moveJ/IK 到固定目标位姿",
            lambda: robot.move_j_ik(
                pick_approach_pose,
                args.joint_speed,
                args.joint_acc,
                args.motion_timeout,
            ),
        )
        confirm_and_run(
            "沿末端 +Z 方向 moveL 前进 {:.1f} mm".format(args.tool_advance_mm),
            lambda: robot.move_l(
                pick_grasp_pose,
                args.linear_speed,
                args.linear_acc,
                args.motion_timeout,
            ),
        )
        confirm_and_run(
            "闭合夹爪抓取白色工件",
            lambda: robot.close_gripper(
                speed=args.gripper_speed,
                force=args.gripper_force,
                timeout_s=args.gripper_timeout,
            ),
        )
        confirm_and_run(
            "moveL 沿原路径退回固定目标位姿",
            lambda: robot.move_l(
                pick_approach_pose,
                args.linear_speed,
                args.linear_acc,
                args.motion_timeout,
            ),
        )
        if args.post_pick_lift > 0.0:
            confirm_and_run(
                "moveL 沿 Base Z 抬升 {:.1f} mm".format(
                    args.post_pick_lift * 1000.0
                ),
                lambda: robot.move_l(
                    post_pick_lift_pose,
                    args.linear_speed,
                    args.linear_acc,
                    args.motion_timeout,
                ),
            )
        confirm_and_run(
            "moveJ/IK 到红盒第 {} 个凹槽上方".format(selected_index + 1),
            lambda: robot.move_j_ik(
                place_pose,
                args.joint_speed,
                args.joint_acc,
                args.motion_timeout,
            ),
        )
        confirm_and_run(
            "moveL 下降到红盒凹槽释放位姿",
            lambda: robot.move_l(
                place_release_pose,
                args.linear_speed,
                args.linear_acc,
                args.motion_timeout,
            ),
        )
        confirm_and_run(
            "打开夹爪释放白色工件",
            lambda: robot.open_gripper(
                speed=args.gripper_speed,
                force=args.gripper_force,
                timeout_s=args.gripper_timeout,
            ),
        )
        print("Fixed-pose pick-and-place sequence completed.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
