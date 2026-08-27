#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/sjh/anaconda3/envs/lerobot_eye_cuda114_clone/bin/python"
FLOW_ARGS=("$@")
set --
source /opt/ros/noetic/setup.bash
source "${PROJECT_ROOT}/catkin_ws/devel/setup.bash"
export YOLO_CONFIG_DIR="/tmp/ultralytics-two-hand"
export MPLCONFIGDIR="/tmp/matplotlib-two-hand"
exec "${PYTHON}" -u \
  "${PROJECT_ROOT}/complete_process/two_hand_pick_and_place.py" "${FLOW_ARGS[@]}"
