# CoRL 2020: Learn to Calibrate

## 1. Introduction
### 1.1 Project description:
Since visual-inertial systems have been prevailed in a wide range of applications, precise calibration is of great importance. Typically, it requires performing sophisticated motion primitives in front of a calibration target. In this project, instead of performing this task manually and empirically, our goal is to apply reinforcement learning to learn the best motion primitives for enough calibration precision. With this result, we aim to achieve automatic calibration of an arbitrary visual-inertial system using a robotic arm.

### 1.2 Build Instructions for Ubuntu:

*Install required dependencies:*
```
sudo apt-get install ros-melodic-moveit
sudo apt install ros-melodic-libfranka ros-melodic-franka-ros
sudo apt-get install ros-kinetic-moveit-visual-tools
sudo apt-get install build-essential bc curl ca-certificates fakeroot gnupg2 libssl-dev lsb-release libelf-dev bison flex
```

*Clone the repository and catkin build:*
```
cd ~/catkin_ws/src
git clone https://github.com/clthegoat/Learn-to-Calibrate.git
cd ../
catkin build
source ~/catkin_ws/devel/setup.bash
```
(if you fail in this step, try to find another computer with clean system or reinstall Ubuntu and ROS)

### 1.3 Examples:
```
roslaunch franka_cal_sim action_srv_nodelet.launch
```

### 1.4 Code framework:
<img src="support_file/img/Selection_010.png" width = 100% height = 100% div align=left />


## 2 Development
### 2.1 Timeline:
* April 19: Midterm progress - Students submit their progress report.
* June 08: Final project presentations - Students present their projects in a joint poster session in CLA Building D-floor (12:00 to 14:00).
* June 14: Final project reports - Students submit their final reports for the projects.
* July 07: CoRL-2020 submission deadline.

### 2.2 Change log:
* 20200404: Fix the simulation

### 2.3 To do:
* state estimation: Camera intrinsic 

### 2.4 Useful links:
* [Github repo: franka_gazebo](https://github.com/mkrizmancic/franka_gazebo.git)
* [Blog: Integrating FRANKA EMIKA Panda robot into Gazebo](https://erdalpekel.de/?p=55)
* [Github repo: panda_simulation](https://github.com/erdalpekel/panda_simulation.git)
* [Github repo: ROS integration for Franka Emika research robots](https://github.com/frankaemika/franka_ros.git)
* [Can I simulate franka panda in Gazebo #44](https://github.com/frankaemika/franka_ros/issues/44)
* [DE3-Panda-Wall: Running on Gazebo](https://de3-panda-wall.readthedocs.io/en/latest/gazebo_problems.html)
