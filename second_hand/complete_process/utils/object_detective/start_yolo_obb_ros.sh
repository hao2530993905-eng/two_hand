#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/sjh/second_hand"
CONDA_PYTHON="/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python"
DETECTOR_ARGS=("$@")
set --

source /opt/ros/noetic/setup.bash
if [[ -f "${PROJECT_ROOT}/catkin_ws/devel/setup.bash" ]]; then
  source "${PROJECT_ROOT}/catkin_ws/devel/setup.bash"
fi

export YOLO_CONFIG_DIR="/tmp/ultralytics-second-hand"
export MPLCONFIGDIR="/tmp/matplotlib-manual-white-obb"

exec "${CONDA_PYTHON}" \
  "${PROJECT_ROOT}/complete_process/utils/object_detective/yolo_obb_ros_detector.py" \
  "${DETECTOR_ARGS[@]}"
