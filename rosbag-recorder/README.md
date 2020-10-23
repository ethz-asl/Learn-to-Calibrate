## rosbag_recorder

ROS service to remotely start/stop recording selected topics to a named bag file.
Includes `*.srv` definitions necessary to issue requests.

## Building

 - Clone the repo into the `src` directory of your catkin workspace
 - Run `catkin_make` from the root of your catkin workspace
 - Run `catkin_make -DCMAKE_INSTALL_PREFIX=/opt/ros/<distro> install` or simply source `devel/setup.bash` from the root of your catkin workspace and copy `devel/include/rosbag_recorder` to the `include` folder of your ROS installation
 
## Running

 - Start the service: `rosrun rosbag_recorder rosbag_recorder_server.py`
 - Use the `qml-ros-recorder` QML plugin in an application to start/stop recording remotely
