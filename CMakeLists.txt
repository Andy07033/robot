cmake_minimum_required(VERSION 3.8)
project(robot)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch meshes rviz urdf
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/joint_state_slider_gui.py
    scripts/zero_joint_state_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
