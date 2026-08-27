cmake_minimum_required(VERSION 3.0.2)
project(d435_eye_on_hand)

find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/camera_info_override.py
  scripts/import_intrinsics.py
  scripts/check_intrinsics.py
  scripts/check_handeye_inputs.py
  scripts/evaluate_handeye_error.py
  scripts/marker_to_base.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
