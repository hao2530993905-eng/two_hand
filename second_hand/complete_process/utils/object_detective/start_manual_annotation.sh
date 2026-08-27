#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ANNOTATOR_ARGS=("$@")
set --

source /opt/ros/noetic/setup.bash
if [[ -f "${PROJECT_ROOT}/catkin_ws/devel/setup.bash" ]]; then
  source "${PROJECT_ROOT}/catkin_ws/devel/setup.bash"
fi

exec /usr/bin/python3 "${SCRIPT_DIR}/manual_dataset_annotator.py" "${ANNOTATOR_ARGS[@]}"
