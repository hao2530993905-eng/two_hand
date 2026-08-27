#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH_ARGS=("$@")
set --
source /opt/ros/noetic/setup.bash
source "${PROJECT_ROOT}/catkin_ws/devel/setup.bash"
exec roslaunch two_hand_bringup both_cameras.launch "${LAUNCH_ARGS[@]}"
