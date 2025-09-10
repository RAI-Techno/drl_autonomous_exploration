cmake_minimum_required(VERSION 3.0.2)
project(aslam)


find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES aslam
#  CATKIN_DEPENDS rospy std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)


catkin_install_python(PROGRAMS
   src/aslam/rainbow/train_model.py
   src/aslam/rainbow/test_model.py
   src/aslam/openai_ros/task_envs/lilybot/lilybot_world.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
 )
