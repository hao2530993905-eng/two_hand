#!/usr/bin/env python3
"""Continuously pick white workpieces and fill unique red-board slots.

The red depth rectangle is acquired once.  After each successful placement the
used slot is removed from consideration, the robot retreats vertically, and
then returns to the TCP pose recorded at startup.  White detections whose
centers lie inside the red rectangle are rejected so already placed parts are
not picked again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
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
from complete_process import visual_pick_and_place as single


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously place up to four YOLO-detected white workpieces in "
            "unique depth-detected red slots"
        )
    )
    parser.add_argument("--white-model", type=Path, default=single.DEFAULT_WHITE_MODEL)
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
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--stable-samples", type=int, default=12)
    parser.add_argument("--association-px", type=float, default=80.0)
    parser.add_argument("--max-sync-delta", type=float, default=0.10)
    parser.add_argument("--depth-radius", type=int, default=5)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument(
        "--first-white-timeout",
        type=float,
        default=30.0,
        help="seconds to wait for the first white workpiece",
    )
    parser.add_argument(
        "--next-white-timeout",
        type=float,
        default=10.0,
        help="stop after this many seconds without the next white workpiece",
    )

    parser.add_argument(
        "--depth-result-topic",
        default="/depth_background_rect_detector/result",
    )
    parser.add_argument("--depth-detection-timeout", type=float, default=30.0)
    parser.add_argument("--red-min-area", type=float, default=3000.0)
    parser.add_argument(
        "--depth-detector-python", type=Path, default=Path("/usr/bin/python3")
    )
    parser.add_argument(
        "--no-auto-start-depth-detector",
        dest="auto_start_depth_detector",
        action="store_false",
    )
    parser.set_defaults(auto_start_depth_detector=True)
    parser.add_argument(
        "--red-exclusion-margin-px",
        type=float,
        default=20.0,
        help="also reject white centers this many pixels outside the red OBB",
    )

    parser.add_argument(
        "--long-fractions",
        nargs=4,
        type=float,
        default=(11.0 / 64.0, 24.0 / 64.0, 39.0 / 64.0, 52.0 / 64.0),
        metavar=("F1", "F2", "F3", "F4"),
    )
    parser.add_argument("--width-fraction", type=float, default=16.0 / 32.0)
    parser.add_argument("--max-placements", type=int, default=4)

    parser.add_argument("--pick-approach-height", type=float, default=0.10)
    parser.add_argument("--pick-descent", type=float, default=0.06)
    parser.add_argument("--post-pick-lift", type=float, default=0.15)
    parser.add_argument("--place-approach-height", type=float, default=0.15)
    parser.add_argument("--place-descent", type=float, default=0.08)
    parser.add_argument(
        "--workspace",
        nargs=6,
        type=float,
        default=(-0.9, 0.9, 0.20, 0.95, 0.02, 0.65),
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
    )
    parser.add_argument("--joint-speed", type=float, default=0.10)
    parser.add_argument("--joint-acc", type=float, default=0.20)
    parser.add_argument("--linear-speed", type=float, default=0.01)
    parser.add_argument("--linear-acc", type=float, default=0.03)
    parser.add_argument("--motion-timeout", type=float, default=60.0)
    parser.add_argument("--gripper-timeout", type=float, default=5.0)
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=PROJECT_ROOT / "continuous_placement_debug",
    )
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--execute", action="store_true")
    single.add_white_edge_arguments(parser)
    args = parser.parse_args(rospy.myargv()[1:])

    if not args.white_model.is_file():
        parser.error("model does not exist: {}".format(args.white_model))
    if not args.depth_detector_python.is_file():
        parser.error(
            "depth detector Python does not exist: {}".format(
                args.depth_detector_python
            )
        )
    if not 0.0 < args.confidence <= 1.0:
        parser.error("--confidence must be in (0,1]")
    if args.red_min_area < 0.0:
        parser.error("--red-min-area must be non-negative")
    if args.red_exclusion_margin_px < 0.0:
        parser.error("--red-exclusion-margin-px must be non-negative")
    if args.max_placements < 1 or args.max_placements > 4:
        parser.error("--max-placements must be in [1,4]")
    if len(set(args.long_fractions)) != 4 or any(
        value < 0.0 or value > 1.0 for value in args.long_fractions
    ):
        parser.error("--long-fractions must contain four unique values in [0,1]")
    if not 0.0 <= args.width_fraction <= 1.0:
        parser.error("--width-fraction must be in [0,1]")
    if not 0 <= args.gripper_speed <= 255:
        parser.error("--gripper-speed must be in [0,255]")
    if not 0 <= args.gripper_force <= 255:
        parser.error("--gripper-force must be in [0,255]")
    positive = (
        "first_white_timeout",
        "next_white_timeout",
        "depth_detection_timeout",
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
    )
    for name in positive:
        if getattr(args, name) <= 0.0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if args.pick_descent >= args.pick_approach_height:
        parser.error("--pick-descent must be less than --pick-approach-height")
    if args.place_descent >= args.place_approach_height:
        parser.error("--place-descent must be less than --place-approach-height")
    single.validate_white_edge_arguments(parser, args)
    return args


def prompt_cycle(cycle: int, slot_number: int) -> bool:
    while True:
        try:
            answer = input(
                "Cycle {} -> slot {}: press Enter to execute, q to quit: ".format(
                    cycle, slot_number
                )
            ).strip().lower()
        except EOFError:
            raise RuntimeError("terminal input is unavailable")
        if answer == "":
            return True
        if answer == "q":
            return False
        print("Only Enter or q is accepted.")


def acquire_white(
    acquirer: vision.YoloTargetAcquirer,
    timeout: float,
    cycle: int,
) -> Optional[Dict[str, Any]]:
    acquirer.args.input_timeout = timeout
    rospy.loginfo(
        "Waiting up to %.1f s for white workpiece %d...", timeout, cycle
    )
    try:
        target = acquirer.acquire()
    except TimeoutError:
        rospy.loginfo("No new white workpiece detected within %.1f s", timeout)
        return None
    rospy.loginfo(
        "White workpiece %d acquired: confidence=%.4f center=[%.2f, %.2f]",
        cycle,
        target["confidence"],
        target["center_px"][0],
        target["center_px"][1],
    )
    return target


def install_red_box_exclusion(
    acquirer: vision.YoloTargetAcquirer,
    red_corners_px: Sequence[Sequence[float]],
    margin_px: float,
) -> None:
    """Reject white detections inside/near the red OBB after first placement."""
    raw_infer = acquirer.infer
    contour = np.asarray(red_corners_px, dtype=np.float32).reshape(-1, 1, 2)

    def infer_outside_red(image: np.ndarray) -> List[Dict[str, Any]]:
        objects = raw_infer(image)
        accepted = []
        rejected = 0
        for item in objects:
            center = tuple(map(float, item["center_px"]))
            signed_distance = cv2.pointPolygonTest(contour, center, True)
            if signed_distance >= -margin_px:
                rejected += 1
            else:
                accepted.append(item)
        if rejected:
            rospy.loginfo_throttle(
                2.0,
                "Ignoring %d white detection(s) in the red box",
                rejected,
            )
        return accepted

    acquirer.infer = infer_outside_red


def compute_red_slots(
    red_center: np.ndarray,
    red_short: np.ndarray,
    red_long: np.ndarray,
    red_width: float,
    red_length: float,
    long_fractions: Sequence[float],
    width_fraction: float,
) -> List[np.ndarray]:
    slots = single.ordered_fractional_points(
        red_center, red_long, red_length, long_fractions
    )
    offset = (float(width_fraction) - 0.5) * red_width * red_short
    return [slot + offset for slot in slots]


def select_nearest_unused_slot(
    white_center: np.ndarray,
    slots: Sequence[np.ndarray],
    used_slots: Set[int],
) -> Tuple[int, List[float]]:
    distances = [
        float(np.linalg.norm(np.asarray(slot)[:2] - white_center[:2]))
        for slot in slots
    ]
    available = [index for index in range(len(slots)) if index not in used_slots]
    if not available:
        raise RuntimeError("all red slots are already occupied")
    selected = min(available, key=lambda index: distances[index])
    return selected, distances


def latest_color_message(collector: vision.SensorCollector) -> Image:
    with collector.condition:
        message = collector.color_message
    if message is None:
        raise RuntimeError("color image is unavailable")
    return message


def publish_cycle_debug(
    collector: vision.SensorCollector,
    red: Dict[str, Any],
    slots: Sequence[np.ndarray],
    long_fractions: Sequence[float],
    width_fraction: float,
    white: Dict[str, Any],
    selected_index: int,
    distances: Sequence[float],
    used_slots: Set[int],
    camera_info: Any,
    base_from_camera: np.ndarray,
    publisher: Any,
    output_path: Path,
) -> None:
    color_message = latest_color_message(collector)
    debug = vision.color_message_to_bgr(color_message)
    corners = np.rint(np.asarray(red["corners_px"])).astype(np.int32)
    cv2.polylines(debug, [corners], True, (0, 255, 0), 4, cv2.LINE_AA)
    cv2.putText(
        debug,
        "RED DEPTH OBB",
        (max(10, int(corners[:, 0].min())), max(30, int(corners[:, 1].min()) - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    white_px = tuple(np.rint(np.asarray(white["center_px"])).astype(np.int32))
    white_rectangle = np.rint(
        np.asarray(white["corners_px"], dtype=np.float64)
    ).astype(np.int32)
    white_yolo = np.rint(
        np.asarray(white["yolo_corners_px"], dtype=np.float64)
    ).astype(np.int32)
    white_contour = np.rint(
        np.asarray(white["detailed_contour_px"], dtype=np.float64)
    ).astype(np.int32)
    cv2.polylines(debug, [white_yolo], True, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.polylines(
        debug, [white_contour], True, (255, 0, 255), 3, cv2.LINE_AA
    )
    cv2.polylines(
        debug, [white_rectangle], True, (255, 0, 0), 4, cv2.LINE_AA
    )
    cv2.drawMarker(debug, white_px, (255, 0, 0), cv2.MARKER_CROSS, 28, 3, cv2.LINE_AA)
    cv2.putText(
        debug,
        "WHITE EDGE + INNER RECT",
        (white_px[0] + 12, white_px[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    slot_pixels = [
        single.base_point_to_color_pixel(slot, camera_info, base_from_camera)
        for slot in slots
    ]
    selected_px = tuple(np.rint(slot_pixels[selected_index]).astype(np.int32))
    cv2.line(debug, white_px, selected_px, (0, 255, 255), 3, cv2.LINE_AA)
    for index, pixel in enumerate(slot_pixels):
        location = tuple(np.rint(pixel).astype(np.int32))
        if index in used_slots:
            color, label = (100, 100, 100), "USED"
            cv2.drawMarker(
                debug, location, color, cv2.MARKER_TILTED_CROSS, 30, 4, cv2.LINE_AA
            )
        elif index == selected_index:
            color, label = (0, 255, 255), "SELECTED"
            cv2.drawMarker(
                debug, location, color, cv2.MARKER_CROSS, 34, 4, cv2.LINE_AA
            )
        else:
            color, label = (255, 255, 0), "AVAILABLE"
            cv2.circle(debug, location, 9, color, 3, cv2.LINE_AA)
        cv2.putText(
            debug,
            "S{} L={:.3f} W={:.3f} {:.3f}m {}".format(
                index + 1,
                long_fractions[index],
                width_fraction,
                distances[index],
                label,
            ),
            (location[0] + 12, location[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

    publisher.publish(vision.bgr_to_image_message(debug, color_message.header))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), debug):
        rospy.logwarn("Could not save cycle debug image: %s", output_path)


def print_cycle_plan(
    cycle: int,
    white_center: np.ndarray,
    white_pick_surface: np.ndarray,
    selected_index: int,
    slots: Sequence[np.ndarray],
    distances: Sequence[float],
    used_slots: Set[int],
    white_approach_pose: Sequence[float],
    white_grasp_pose: Sequence[float],
    place_pose: Sequence[float],
    release_pose: Sequence[float],
    home_pose: Sequence[float],
) -> None:
    print("\n" + "=" * 78)
    print("CONTINUOUS CYCLE {} PLAN".format(cycle))
    print("white center base_link:", white_center.round(6).tolist())
    print("white pick surface base_link:", white_pick_surface.round(6).tolist())
    print("used slots before this cycle:", [index + 1 for index in sorted(used_slots)])
    for index, (slot, distance) in enumerate(zip(slots, distances)):
        state = "USED" if index in used_slots else "AVAILABLE"
        if index == selected_index:
            state = "SELECTED (nearest unused)"
        print(
            "slot {}: distance_to_white={:.4f} m state={} base_link={}".format(
                index + 1, distance, state, slot.round(6).tolist()
            )
        )
    print("selected slot:", selected_index + 1)
    single.print_pose("white approach UR base", white_approach_pose)
    single.print_pose("white grasp UR base", white_grasp_pose)
    single.print_pose("red approach UR base", place_pose)
    single.print_pose("red release UR base", release_pose)
    single.print_pose("return-to-initial UR base", home_pose)
    print("=" * 78)


def main() -> None:
    args = parse_args()
    rospy.init_node("continuous_visual_pick_and_place")
    collector = vision.SensorCollector(
        args.color_topic, args.depth_topic, args.camera_info_topic
    )
    args.model = args.white_model
    # YoloTargetAcquirer expects input_timeout to exist and it is changed per cycle.
    args.input_timeout = args.first_white_timeout
    acquirer = single.WhiteEdgeTargetAcquirer(args, collector)
    base_from_camera = single.wait_for_base_camera_matrix(args)
    low, high = vision.parse_bounds(args.workspace)

    result_pub = rospy.Publisher("~cycle_result", String, queue_size=1, latch=True)
    debug_pub = rospy.Publisher(
        "~placement_debug_image", Image, queue_size=1, latch=True
    )
    print("\nCamera recognition topics:")
    print("  white YOLO+edge: /continuous_visual_pick_and_place/debug_image")
    print("  red depth:  /depth_background_rect_detector/debug_image")
    print(
        "  combined:   "
        "/continuous_visual_pick_and_place/placement_debug_image\n"
    )

    robot = None
    home_pose = single.GRIPPER_BASE_X_REFERENCE_ROTVEC.tolist()
    home_tcp_pose: List[float]
    if args.execute:
        robot = single.UR_BASE(args.host, gripper_port=args.gripper_port)
        connected = robot.connect()
        if not connected.success:
            raise RuntimeError(connected.message)
        current_pose = vision.verify_robot_ready(robot)
        home_tcp_pose = list(map(float, current_pose))
        home_rotation = np.asarray(home_tcp_pose[3:], dtype=np.float64)
    else:
        home_tcp_pose = [0.0, 0.0, 0.0] + home_pose
        home_rotation = np.asarray(home_pose, dtype=np.float64)

    red: Optional[Dict[str, Any]] = None
    red_slots: List[np.ndarray] = []
    red_long: Optional[np.ndarray] = None
    used_slots: Set[int] = set()
    cycle = 1

    try:
        while not rospy.is_shutdown() and len(used_slots) < args.max_placements:
            timeout = (
                args.first_white_timeout if cycle == 1 else args.next_white_timeout
            )
            white = acquire_white(acquirer, timeout, cycle)
            if white is None:
                break

            white_geometry = single.robust_white_pick_geometry(
                white,
                61.0 / 128.0,
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

            if red is None:
                single.ensure_depth_detector(args)
                rospy.loginfo("Acquiring the red depth rectangle once...")
                red = single.acquire_depth_red_target(args, collector)
                red_surface = single.base_surface_point(red, base_from_camera)
                (
                    red_center,
                    red_short,
                    red_long,
                    red_width,
                    red_length,
                ) = single.prepare_target_geometry(
                    red, red_surface, collector, base_from_camera
                )
                red_slots = compute_red_slots(
                    red_center,
                    red_short,
                    red_long,
                    red_width,
                    red_length,
                    args.long_fractions,
                    args.width_fraction,
                )
                install_red_box_exclusion(
                    acquirer, red["corners_px"], args.red_exclusion_margin_px
                )
                rospy.loginfo("Red geometry cached; four unique slots are ready")

            selected_index, distances = select_nearest_unused_slot(
                white_center, red_slots, used_slots
            )
            selected_slot = red_slots[selected_index]

            white_approach_xyz = white_pick_surface.copy()
            white_approach_xyz[2] += args.pick_approach_height
            white_grasp_xyz = white_approach_xyz.copy()
            white_grasp_xyz[2] -= args.pick_descent
            post_pick_xyz = white_grasp_xyz.copy()
            post_pick_xyz[2] += args.post_pick_lift

            place_approach_xyz = selected_slot.copy()
            place_approach_xyz[2] += args.place_approach_height
            place_release_xyz = place_approach_xyz.copy()
            place_release_xyz[2] -= args.place_descent
            for name, point in (
                ("white approach", white_approach_xyz),
                ("white grasp", white_grasp_xyz),
                ("post-pick lift", post_pick_xyz),
                ("red approach", place_approach_xyz),
                ("red release", place_release_xyz),
            ):
                single.validate_workspace(point, low, high, name)

            white_angle = single.base_link_axis_angle_in_ur_base(white_short)
            white_rotation = single.aligned_gripper_rotvec(
                white_angle, home_rotation
            )
            place_angle = single.base_link_axis_angle_in_ur_base(red_long)
            place_rotation = single.aligned_gripper_rotvec(
                place_angle, white_rotation
            )
            white_approach_pose = single.make_ur_pose(
                white_approach_xyz, white_rotation
            )
            white_grasp_pose = single.make_ur_pose(white_grasp_xyz, white_rotation)
            post_pick_pose = single.make_ur_pose(post_pick_xyz, white_rotation)
            place_pose = single.make_ur_pose(place_approach_xyz, place_rotation)
            release_pose = single.make_ur_pose(place_release_xyz, place_rotation)

            print_cycle_plan(
                cycle,
                white_center,
                white_pick_surface,
                selected_index,
                red_slots,
                distances,
                used_slots,
                white_approach_pose,
                white_grasp_pose,
                place_pose,
                release_pose,
                home_tcp_pose,
            )
            debug_path = args.debug_dir.expanduser().resolve() / (
                "cycle_{:02d}_slot_{}.png".format(cycle, selected_index + 1)
            )
            publish_cycle_debug(
                collector,
                red,
                red_slots,
                args.long_fractions,
                args.width_fraction,
                white,
                selected_index,
                distances,
                used_slots,
                single.latest_camera_info(collector),
                base_from_camera,
                debug_pub,
                debug_path,
            )
            print("debug image:", debug_path)
            print(
                "ROS debug topic: "
                "/continuous_visual_pick_and_place/placement_debug_image"
            )

            payload = {
                "cycle": cycle,
                "white_center_base_link_m": white_center.tolist(),
                "selected_slot": selected_index + 1,
                "selected_slot_base_link_m": selected_slot.tolist(),
                "distance_to_white_m": distances[selected_index],
                "white_robust_samples_total": white_geometry["total_samples"],
                "white_robust_samples_used": white_geometry["inlier_samples"],
                "white_robust_samples_rejected": white_geometry["rejected_samples"],
                "white_robust_rejection_limit_mm": white_geometry[
                    "rejection_limit_mm"
                ],
                "used_slots_before": [index + 1 for index in sorted(used_slots)],
                "white_approach_ur_pose": white_approach_pose,
                "white_grasp_ur_pose": white_grasp_pose,
                "red_approach_ur_pose": place_pose,
                "red_release_ur_pose": release_pose,
                "home_ur_pose": home_tcp_pose,
            }
            result_pub.publish(String(data=json.dumps(payload)))

            if not args.execute:
                print("DRY RUN: no robot motion executed; stopping after cycle 1.")
                break
            if not prompt_cycle(cycle, selected_index + 1):
                print("q received before motion; continuous workflow stopped safely.")
                break

            acquirer.start_live_preview(white)
            try:
                single.run_motion(
                    "cycle {} open gripper".format(cycle),
                    lambda: robot.open_gripper(
                        speed=args.gripper_speed,
                        force=args.gripper_force,
                        timeout_s=args.gripper_timeout,
                    ),
                )
                single.run_motion(
                    "cycle {} move to white approach".format(cycle),
                    lambda: robot.move_j_ik(
                        white_approach_pose,
                        args.joint_speed,
                        args.joint_acc,
                        args.motion_timeout,
                    ),
                )
                single.run_motion(
                    "cycle {} descend to white grasp".format(cycle),
                    lambda: robot.move_l(
                        white_grasp_pose,
                        args.linear_speed,
                        args.linear_acc,
                        args.motion_timeout,
                    ),
                )
                single.run_motion(
                    "cycle {} close gripper".format(cycle),
                    lambda: robot.close_gripper(
                        speed=args.gripper_speed,
                        force=args.gripper_force,
                        timeout_s=args.gripper_timeout,
                    ),
                )
                single.run_motion(
                    "cycle {} lift white workpiece".format(cycle),
                    lambda: robot.move_l(
                        post_pick_pose,
                        args.linear_speed,
                        args.linear_acc,
                        args.motion_timeout,
                    ),
                )
            finally:
                acquirer.stop_live_preview()

            single.run_motion(
                "cycle {} move to red approach".format(cycle),
                lambda: robot.move_j_ik(
                    place_pose,
                    args.joint_speed,
                    args.joint_acc,
                    args.motion_timeout,
                ),
            )
            single.run_motion(
                "cycle {} descend into red slot {}".format(
                    cycle, selected_index + 1
                ),
                lambda: robot.move_l(
                    release_pose,
                    args.linear_speed,
                    args.linear_acc,
                    args.motion_timeout,
                ),
            )
            single.run_motion(
                "cycle {} release white workpiece".format(cycle),
                lambda: robot.open_gripper(
                    speed=args.gripper_speed,
                    force=args.gripper_force,
                    timeout_s=args.gripper_timeout,
                ),
            )
            used_slots.add(selected_index)
            single.run_motion(
                "cycle {} retreat vertically from red box".format(cycle),
                lambda: robot.move_l(
                    place_pose,
                    args.linear_speed,
                    args.linear_acc,
                    args.motion_timeout,
                ),
            )
            single.run_motion(
                "cycle {} return to initial pose".format(cycle),
                lambda: robot.move_j_ik(
                    home_tcp_pose,
                    args.joint_speed,
                    args.joint_acc,
                    args.motion_timeout,
                ),
            )
            rospy.loginfo(
                "Cycle %d completed; occupied slots: %s",
                cycle,
                [index + 1 for index in sorted(used_slots)],
            )
            cycle += 1

        print(
            "Continuous workflow finished: {} placement(s), occupied slots={}".format(
                len(used_slots), [index + 1 for index in sorted(used_slots)]
            )
        )
    finally:
        acquirer.stop_live_preview()
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    main()
