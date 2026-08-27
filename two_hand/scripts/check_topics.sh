#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/noetic/setup.bash
source "${PROJECT_ROOT}/catkin_ws/devel/setup.bash"

for topic in \
  /d435/color/image_raw \
  /d435/aligned_depth_to_color/image_raw \
  /d435/color/camera_info \
  /camera/color/image_raw \
  /camera/aligned_depth_to_color/image_raw \
  /camera/color/camera_info; do
  timeout 6 rostopic echo -n 1 "${topic}" >/dev/null || {
    echo "ERROR: topic unavailable: ${topic}" >&2
    exit 1
  }
  echo "OK: ${topic}"
done
