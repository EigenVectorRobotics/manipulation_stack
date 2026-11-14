cmake_minimum_required(VERSION 3.22...3.30)
project(humanoid_manipulation_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/ControlJointTrajectory.action"
  "action/MoveToPose.action"
  "action/VisualServoingArm.action"
  "srv/PlanJointTrajectory.srv"
  "srv/CheckReach.srv"
  DEPENDENCIES geometry_msgs trajectory_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
