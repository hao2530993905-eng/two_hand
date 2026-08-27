cmake_minimum_required(VERSION 3.0.2)
project(two_hand_bringup)
find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs)
catkin_package()
catkin_install_python(PROGRAMS scripts/camera_info_override.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
