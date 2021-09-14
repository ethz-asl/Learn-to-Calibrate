# generated from catkin/cmake/template/pkg.context.pc.in
CATKIN_PACKAGE_PREFIX = ""
PROJECT_PKG_CONFIG_INCLUDE_DIRS = "${prefix}/include".split(';') if "${prefix}/include" != "" else []
PROJECT_CATKIN_DEPENDS = "roscpp;sensor_msgs;gazebo_msgs;geometry_msgs;tf;tf2;message_runtime;std_msgs;random_numbers;std_srvs;nodelet;moveit_msgs;moveit_core;moveit_visual_tools;moveit_ros_planning_interface".replace(';', ' ')
PKG_CONFIG_LIBRARIES_WITH_PREFIX = "-lfranka_cal_sim".split(';') if "-lfranka_cal_sim" != "" else []
PROJECT_NAME = "franka_cal_sim"
PROJECT_SPACE_DIR = "/usr/local"
PROJECT_VERSION = "0.0.0"
