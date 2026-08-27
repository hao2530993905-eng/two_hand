#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREVIEW_ARGS=("$@")
set --
source /opt/ros/noetic/setup.bash
source "${PROJECT_ROOT}/catkin_ws/devel/setup.bash"
exec /usr/bin/python3 -u \
  "${PROJECT_ROOT}/complete_process/utils/object_detective/depth_background_rect_detector.py" \
  _color_topic:=/camera/color/image_raw \
  _depth_topic:=/camera/aligned_depth_to_color/image_raw \
  _background_path:="${PROJECT_ROOT}/complete_process/utils/object_detective/depth_background.npy" \
  _target_color:=red _min_area:=3000 _collect_dataset:=false \
  _show_gui:=false __name:=depth_background_rect_detector "${PREVIEW_ARGS[@]}"
