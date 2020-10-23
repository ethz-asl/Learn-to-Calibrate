cmake_minimum_required(VERSION 2.8.3)
project(nodelet_rosbag)

find_package(catkin REQUIRED COMPONENTS nodelet rosbag_storage roscpp std_msgs actionlib actionlib_msgs topic_tools)

find_package(Boost REQUIRED COMPONENTS thread)

add_action_files(
  DIRECTORY action
  FILES
  Subscribe.action
  Start.action
  Stop.action
  Record.action
)

generate_messages(
  DEPENDENCIES actionlib_msgs std_msgs
)

catkin_package(
  LIBRARIES nodelet_rosbag
  INCLUDE_DIRS include
  CATKIN_DEPENDS nodelet rosbag_storage roscpp actionlib message_generation actionlib_msgs topic_tools
)

include_directories(include ${catkin_INCLUDE_DIRS})

add_library(nodelet_rosbag src/nodelet_rosbag.cpp)

target_link_libraries(nodelet_rosbag ${catkin_LIBRARIES})

add_dependencies(nodelet_rosbag ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

install(TARGETS nodelet_rosbag
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(FILES nodelet_rosbag.xml nodelet_rosbag.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
