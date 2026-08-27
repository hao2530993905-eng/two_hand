from __future__ import annotations

import math
from typing import Callable, Iterable, List


def validated_pose(values: Iterable[float], name: str = "pose") -> List[float]:
    pose = [float(value) for value in values]
    if len(pose) != 6:
        raise ValueError(f"{name} must contain exactly 6 values")
    if not all(math.isfinite(value) for value in pose):
        raise ValueError(f"{name} contains a non-finite value")
    return pose


def offset_base_z(pose: Iterable[float], distance: float) -> List[float]:
    target = validated_pose(pose)
    value = float(distance)
    if not math.isfinite(value):
        raise ValueError("distance must be finite")
    target[2] += value
    return target


def run_step(name: str, action: Callable[[], object]) -> None:
    result = action()
    print(f"{name}: {result}")
    if not result.success:
        raise RuntimeError(f"step failed: {name}: {result.message}")
