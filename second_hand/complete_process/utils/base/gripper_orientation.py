#!/usr/bin/env python3
"""Orientation-only calibration between the UR TCP and gripper finger line.

The reference TCP orientation was measured while the tool was vertical and
the undirected line between the two fingers was parallel to UR controller
Base +X.  A requested finger-line angle is applied around Base Z.  Because a
line is unchanged by 180 degrees, the candidate closer to the current TCP
orientation can be selected to avoid unnecessary wrist motion.
"""

import argparse
import math
from typing import Optional, Sequence

import cv2
import numpy as np


# Measured on 2026-08-26. Position is intentionally excluded: this is an
# orientation reference, not a TCP translation calibration.
GRIPPER_BASE_X_REFERENCE_ROTVEC = np.array(
    [-1.789396, 2.572014, -0.018453], dtype=np.float64
)
REFERENCE_WRIST_3_DEG = 110.0


def rotation_z(angle_rad: float) -> np.ndarray:
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
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
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation matrix must be a finite 3x3 array")
    vector, _ = cv2.Rodrigues(matrix)
    return vector.reshape(3)


def rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


def aligned_gripper_rotvec(
    target_angle_in_ur_base_rad: float,
    nearby_rotvec: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Return a TCP rotvec whose finger line has the requested Base-XY angle.

    ``target_angle_in_ur_base_rad`` is measured counter-clockwise from UR
    controller Base +X toward +Y.  The finger-to-finger line has no direction,
    so angles theta and theta + pi are physically equivalent.  If a nearby TCP
    rotvec is supplied, the closer equivalent is returned.
    """
    if not math.isfinite(target_angle_in_ur_base_rad):
        raise ValueError("target angle must be finite")
    reference = rotvec_to_matrix(GRIPPER_BASE_X_REFERENCE_ROTVEC)
    candidates = [
        rotation_z(target_angle_in_ur_base_rad) @ reference,
        rotation_z(target_angle_in_ur_base_rad + math.pi) @ reference,
    ]
    if nearby_rotvec is None:
        selected = candidates[0]
    else:
        nearby = rotvec_to_matrix(nearby_rotvec)
        selected = min(candidates, key=lambda item: rotation_distance(nearby, item))
    return matrix_to_rotvec(selected)


def finger_line_in_tcp() -> np.ndarray:
    """Return the measured Base +X finger-line direction in TCP coordinates."""
    reference = rotvec_to_matrix(GRIPPER_BASE_X_REFERENCE_ROTVEC)
    return reference.T @ np.array([1.0, 0.0, 0.0], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a desired gripper finger-line angle to a UR TCP rotvec."
    )
    parser.add_argument(
        "--target-angle-deg",
        type=float,
        default=0.0,
        help="undirected finger-line angle in UR Base XY, from +X toward +Y",
    )
    parser.add_argument(
        "--nearby-rotvec",
        type=float,
        nargs=3,
        metavar=("RX", "RY", "RZ"),
        help="current/nearby TCP rotvec used to choose the closer 180-degree solution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = aligned_gripper_rotvec(
        math.radians(args.target_angle_deg), args.nearby_rotvec
    )
    line_tcp = finger_line_in_tcp()
    yaw_offset = math.degrees(math.atan2(line_tcp[1], line_tcp[0]))
    print("reference wrist_3: {:.6f} deg (record only)".format(
        REFERENCE_WRIST_3_DEG
    ))
    print("finger line in TCP: [{}]".format(
        ", ".join("{:.9f}".format(value) for value in line_tcp)
    ))
    print("projected TCP-Z mounting yaw: {:.6f} deg".format(yaw_offset))
    print("target finger-line angle: {:.6f} deg".format(args.target_angle_deg))
    print("target TCP rotvec [rx, ry, rz]: [{}]".format(
        ", ".join("{:.9f}".format(value) for value in result)
    ))


if __name__ == "__main__":
    main()
