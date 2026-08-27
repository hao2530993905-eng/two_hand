#!/usr/bin/env python3
"""Evaluate eye-on-hand calibration by checking the fixed marker pose consistency."""

import argparse
import math
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import rospy
import rospkg
import yaml
from easy_handeye_msgs.srv import ComputeCalibration, SetAlgorithm, TakeSample
from std_srvs.srv import Empty


DEFAULT_NAMESPACE = "/d435_ur3_eye_on_hand"
PARK_ALGORITHM = "OpenCV/Park"


def transform_matrix(transform):
    """Convert geometry_msgs/Transform to a 4x4 parent_T_child matrix."""
    t = transform.translation
    q = transform.rotation
    quat = np.array([q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError("zero-length quaternion")
    x, y, z, w = quat / norm
    rotation = np.array([
        [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w), 2.0 * (x*z + y*w)],
        [2.0 * (x*y + z*w), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w)],
        [2.0 * (x*z - y*w), 2.0 * (y*z + x*w), 1.0 - 2.0 * (x*x + y*y)],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (t.x, t.y, t.z)
    return matrix


def transform_dict(transform):
    return {
        "x": float(transform.translation.x),
        "y": float(transform.translation.y),
        "z": float(transform.translation.z),
        "qx": float(transform.rotation.x),
        "qy": float(transform.rotation.y),
        "qz": float(transform.rotation.z),
        "qw": float(transform.rotation.w),
    }


def average_rotation(rotations):
    """Return the chordal mean rotation, projected back onto SO(3)."""
    mean = np.mean(rotations, axis=0)
    u, _, vt = np.linalg.svd(mean)
    rotation = np.dot(u, vt)
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = np.dot(u, vt)
    return rotation


def rotation_distance_deg(rotation, reference):
    relative = np.dot(reference.T, rotation)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def robust_z_scores(values):
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad < 1e-12:
        scale = max(float(np.std(values)), 1e-12)
        return np.abs(values - median) / scale
    return 0.67448975 * np.abs(values - median) / mad


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "max": float(np.max(values)),
    }


def default_output_dir():
    package_path = rospkg.RosPack().get_path("d435_eye_on_hand")
    return os.path.abspath(os.path.join(package_path, "..", "..", "..", "calibration_results"))


def evaluate(samples, tool_camera):
    robot = samples.hand_world_samples
    marker = samples.camera_marker_samples
    if len(robot) != len(marker):
        raise RuntimeError("robot and marker sample counts differ")
    if len(robot) < 3:
        raise RuntimeError("at least 3 samples are required (20-25 diverse poses are recommended)")

    base_marker = []
    for base_tool_msg, camera_marker_msg in zip(robot, marker):
        base_tool = transform_matrix(base_tool_msg)
        camera_marker = transform_matrix(camera_marker_msg)
        base_marker.append(np.dot(np.dot(base_tool, tool_camera), camera_marker))

    translations = np.array([pose[:3, 3] for pose in base_marker])
    rotations = np.array([pose[:3, :3] for pose in base_marker])
    # Median is less sensitive to a bad pose; rotation uses an SO(3) chordal mean.
    center_translation = np.median(translations, axis=0)
    center_rotation = average_rotation(rotations)
    translation_mm = np.linalg.norm(translations - center_translation, axis=1) * 1000.0
    rotation_deg = np.array([
        rotation_distance_deg(rotation, center_rotation) for rotation in rotations
    ])

    translation_z = robust_z_scores(translation_mm)
    rotation_z = robust_z_scores(rotation_deg)
    combined_z = np.maximum(translation_z, rotation_z)
    statistical = np.where(combined_z > 3.5)[0].tolist()
    worst = int(np.argmax(combined_z))
    # The user requested at least one candidate every run. A candidate is not
    # automatically a bad sample and is never deleted by this script.
    candidates = statistical if statistical else [worst]

    rows = []
    for index, (translation, rotation, score) in enumerate(
            zip(translation_mm, rotation_deg, combined_z)):
        rows.append({
            "sample_index_zero_based": index,
            "sample_number_one_based": index + 1,
            "translation_error_mm": float(translation),
            "rotation_error_deg": float(rotation),
            "robust_score": float(score),
            "statistical_outlier": index in statistical,
            "outlier_candidate": index in candidates,
        })

    return {
        "sample_count": len(rows),
        "translation_error_mm": summarize(translation_mm),
        "rotation_error_deg": summarize(rotation_deg),
        "statistical_outlier_indices_zero_based": statistical,
        "outlier_candidate_indices_zero_based": candidates,
        "candidate_policy": (
            "robust score > 3.5; if none exceeds it, report the maximum-score sample"
        ),
        "samples": rows,
    }


def print_report(report):
    trans = report["translation_error_mm"]
    rot = report["rotation_error_deg"]
    print("\nD435 eye-on-hand residual report")
    print("Samples: {}".format(report["sample_count"]))
    print("Translation [mm]: mean={mean:.3f}, median={median:.3f}, "
          "RMS={rms:.3f}, max={max:.3f}".format(**trans))
    print("Rotation [deg]:   mean={mean:.3f}, median={median:.3f}, "
          "RMS={rms:.3f}, max={max:.3f}".format(**rot))
    print("\n idx(0)  no.(1)   trans_mm   rot_deg   score   candidate")
    for row in report["samples"]:
        mark = "YES" if row["outlier_candidate"] else ""
        print(" {0:>5d}  {1:>6d}  {2:>9.3f}  {3:>8.3f}  {4:>6.2f}   {5}".format(
            row["sample_index_zero_based"], row["sample_number_one_based"],
            row["translation_error_mm"], row["rotation_error_deg"],
            row["robust_score"], mark))
    candidates = report["outlier_candidate_indices_zero_based"]
    print("\nOutlier candidate(s), zero-based index: {}".format(candidates))
    if not report["statistical_outlier_indices_zero_based"]:
        print("No sample exceeded the robust threshold; the worst sample is listed for review.")
    print("The script never removes samples automatically.")


def save_report(report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = os.path.join(output_dir, "d435_handeye_park_error_{}.yaml".format(stamp))
    with open(output_path, "w") as stream:
        yaml.safe_dump(report, stream, default_flow_style=False, sort_keys=False)
    print("Report saved to: {}".format(output_path))
    return output_path


def calibration_path(namespace):
    filename = namespace.rstrip("/").split("/")[-1] + ".yaml"
    return os.path.expanduser(os.path.join("~/.ros/easy_handeye", filename))


def sample_signature(samples):
    """Detect additions, removals, and replacements, not only count changes."""
    values = []
    for sequence in (samples.hand_world_samples, samples.camera_marker_samples):
        for transform in sequence:
            values.extend((
                transform.translation.x, transform.translation.y,
                transform.translation.z, transform.rotation.x,
                transform.rotation.y, transform.rotation.z,
                transform.rotation.w,
            ))
    return len(samples.hand_world_samples), tuple(values)


def backup_calibration(path):
    if not os.path.isfile(path):
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = path + ".bak." + stamp
    shutil.copy2(path, backup_path)
    print("Previous calibration backed up to: {}".format(backup_path))
    return backup_path


def save_latest_and_verify(save_calibration, report, namespace):
    path = calibration_path(namespace)
    save_calibration()
    if not os.path.isfile(path):
        raise RuntimeError("Easy Handeye did not create {}".format(path))
    with open(path, "r") as stream:
        saved = yaml.safe_load(stream)
    parameters = saved.get("parameters", {})
    expected_metadata = {
        "eye_on_hand": True,
        "robot_effector_frame": "tool0",
        "tracking_base_frame": "d435_link",
    }
    for key, expected in expected_metadata.items():
        if parameters.get(key) != expected:
            raise RuntimeError(
                "saved calibration has {}={!r}, expected {!r}".format(
                    key, parameters.get(key), expected))
    saved_transform = saved.get("transformation", {})
    computed = report["computed_transform"]
    keys = ("x", "y", "z", "qx", "qy", "qz", "qw")
    try:
        saved_values = np.array([float(saved_transform[key]) for key in keys])
        computed_values = np.array([float(computed[key]) for key in keys])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("invalid saved calibration: {}".format(error))
    if not np.allclose(saved_values, computed_values, rtol=0.0, atol=1e-10):
        raise RuntimeError("saved matrix does not match the newly computed Park result")
    print("Latest Park calibration saved and verified: {}".format(path))


def compute_and_report(get_samples, set_algorithm, compute_calibration, output_dir,
                       namespace, min_samples):
    samples = get_samples().samples
    sample_count = len(samples.hand_world_samples)
    if sample_count < min_samples:
        print("\nPARK: {}/{} samples; waiting for more samples.".format(
            sample_count, min_samples))
        return sample_count, None

    # The rqt drop-down and this monitor share backend state. Select Park again
    # immediately before every computation so a GUI race cannot change it.
    selected = set_algorithm(PARK_ALGORITHM)
    if not selected.success:
        raise RuntimeError("failed to select {}".format(PARK_ALGORITHM))
    computed = compute_calibration()
    if not computed.valid:
        raise RuntimeError("easy_handeye could not compute a valid Park calibration")

    computed_transform = computed.calibration.transform
    report = evaluate(samples, transform_matrix(computed_transform))
    report["generated_at"] = datetime.now().astimezone().isoformat()
    report["calibration_namespace"] = namespace
    report["algorithm"] = PARK_ALGORITHM
    report["transform_direction"] = "tool0 <- d435_link"
    report["computed_transform"] = transform_dict(computed_transform)
    report["marker_consistency_direction"] = "base_link <- d435_aruco_marker_frame"
    print_report(report)
    save_report(report, output_dir)
    return sample_count, report


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Force OpenCV/Park calibration and evaluate fixed-marker residuals "
            "after each newly added sample"
        ))
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE,
                        help="easy_handeye namespace (default: %(default)s)")
    parser.add_argument("--output-dir", default=None,
                        help="report directory (default: second_hand/calibration_results)")
    parser.add_argument(
        "--watch", action="store_true",
        help="keep running and recompute whenever the sample count changes",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=0.5,
        help="sample-list polling interval in seconds for --watch (default: %(default)s)",
    )
    parser.add_argument(
        "--min-samples", type=int, default=3,
        help="do not compute before this many samples exist (default: %(default)s)",
    )
    parser.add_argument(
        "--save-latest", action="store_true",
        help=(
            "after a successful computation, call Easy Handeye Save; in --watch "
            "mode this overwrites the canonical calibration after every sample"
        ),
    )
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])
    if args.poll_interval <= 0.0:
        parser.error("--poll-interval must be positive")
    if args.min_samples < 3:
        parser.error("--min-samples must be at least 3")
    return args


def main():
    args = parse_args()
    rospy.init_node("d435_handeye_error_evaluator", anonymous=True)
    namespace = "/" + args.namespace.strip("/")
    get_samples_name = namespace + "/get_sample_list"
    set_algorithm_name = namespace + "/set_algorithm"
    compute_name = namespace + "/compute_calibration"
    save_name = namespace + "/save_calibration"
    rospy.loginfo(
        "Waiting for Park calibration services: %s, %s and %s",
        get_samples_name, set_algorithm_name, compute_name,
    )
    rospy.wait_for_service(get_samples_name, timeout=10.0)
    rospy.wait_for_service(set_algorithm_name, timeout=10.0)
    rospy.wait_for_service(compute_name, timeout=10.0)

    output_dir = args.output_dir or default_output_dir()
    get_samples = rospy.ServiceProxy(get_samples_name, TakeSample)
    set_algorithm = rospy.ServiceProxy(set_algorithm_name, SetAlgorithm)
    compute_calibration = rospy.ServiceProxy(compute_name, ComputeCalibration)
    save_calibration = None
    backup_done = False
    if args.save_latest:
        rospy.wait_for_service(save_name, timeout=10.0)
        save_calibration = rospy.ServiceProxy(save_name, Empty)

    selected = set_algorithm(PARK_ALGORITHM)
    if not selected.success:
        raise RuntimeError("failed to select {}".format(PARK_ALGORITHM))
    rospy.loginfo("Forced Easy Handeye algorithm to %s", PARK_ALGORITHM)

    if not args.watch:
        _, report = compute_and_report(
            get_samples, set_algorithm, compute_calibration, output_dir, namespace,
            args.min_samples)
        if report is None:
            raise RuntimeError("not enough samples for Park calibration")
        if save_calibration is not None:
            backup_calibration(calibration_path(namespace))
            save_latest_and_verify(save_calibration, report, namespace)
        return

    rospy.loginfo(
        "Watching samples; each addition, removal or replacement will recompute "
        "with OpenCV/Park. "
        "Ctrl+C stops the monitor."
    )
    last_signature = None
    while not rospy.is_shutdown():
        samples = get_samples().samples
        sample_count = len(samples.hand_world_samples)
        signature = sample_signature(samples)
        if signature != last_signature:
            last_signature = signature
            try:
                _, report = compute_and_report(
                    get_samples, set_algorithm, compute_calibration, output_dir,
                    namespace, args.min_samples,
                )
                if report is not None and save_calibration is not None:
                    if not backup_done:
                        backup_calibration(calibration_path(namespace))
                        backup_done = True
                    save_latest_and_verify(save_calibration, report, namespace)
            except (rospy.ServiceException, RuntimeError, ValueError,
                    np.linalg.LinAlgError) as error:
                # Early or poorly distributed poses can make Park singular.
                # Keep watching so the next, more diverse pose can recover it.
                rospy.logerr("Park evaluation failed at %d samples: %s",
                             sample_count, error)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    try:
        main()
    except (rospy.ROSException, rospy.ServiceException, RuntimeError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        sys.exit(1)
