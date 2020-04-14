#ifndef INCLUDE_SIMULATION_SERVER_H_
#define INCLUDE_SIMULATION_SERVER_H_

#include <franka_cal_sim/actionSrv.h>
#include <gazebo_msgs/GetLinkState.h>
#include <gazebo_msgs/LinkState.h>
#include <gazebo_msgs/ModelState.h>
#include <gazebo_msgs/SetLinkState.h>
#include <gazebo_msgs/SetModelState.h>
#include <geometry_msgs/Pose.h>
#include <math.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <std_srvs/Empty.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_datatypes.h>
#include <cmath>
#include <iostream>
#include <string>

#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>

#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>

#include <moveit_visual_tools/moveit_visual_tools.h>

namespace franka_cal_sim
{
class simulation_server : public nodelet::Nodelet
{
private:
  ros::ServiceClient client;
  ros::ServiceClient client2;
  ros::ServiceServer service;

  double value_;
  bool init;

public:
  simulation_server() : value_(0), init(1)
  {
  }
  virtual void onInit();
  bool model(franka_cal_sim::actionSrv::Request &req, franka_cal_sim::actionSrv::Response &res);
};

}  // namespace franka_cal_sim

#endif /* INCLUDE_SIMULATION_SERVER_H_ */
