cmake_minimum_required(VERSION 3.0.2)
project(ur3_d455_handeye)

find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/camera_info_override.py
  scripts/check_setup.py
  scripts/marker_to_base.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
