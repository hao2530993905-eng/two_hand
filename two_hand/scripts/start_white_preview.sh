#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python"
PREVIEW_ARGS=("$@")
set --
source /opt/ros/noetic/setup.bash
source "${PROJECT_ROOT}/catkin_ws/devel/setup.bash"
export YOLO_CONFIG_DIR="/tmp/ultralytics-two-hand"
export MPLCONFIGDIR="/tmp/matplotlib-two-hand"
exec "${PYTHON}" -u \
  "${PROJECT_ROOT}/complete_process/utils/object_detective/yolo_obb_ros_detector.py" \
  _image_topic:=/d435/color/image_raw \
  _model:="${PROJECT_ROOT}/complete_process/utils/object_detective/models/manual_white_rectangle_best.pt" \
  _conf:=0.25 _iou:=0.45 _imgsz:=640 _device:=cpu \
  __name:=white_rectangle_yolo_obb "${PREVIEW_ARGS[@]}"
