#ifndef INCLUDE_SIMULATION_SERVER_H_
#define INCLUDE_SIMULATION_SERVER_H_

#include <franka_cal_sim/RLSrv.h>
#include <franka_cal_sim/actionSrv.h>
#include <franka_cal_sim/estimateSrv.h>
#include <franka_cal_sim/recordSrv.h>
#include <gazebo_msgs/GetLinkState.h>
#include <gazebo_msgs/LinkState.h>
#include <gazebo_msgs/ModelState.h>
#include <gazebo_msgs/SetLinkState.h>
#include <gazebo_msgs/SetModelState.h>
#include <geometry_msgs/Pose.h>
#include <math.h>
#include <nodelet/nodelet.h>
#include <random_numbers/random_numbers.h>
#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <std_srvs/Empty.h>
#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_datatypes.h>
#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace franka_cal_sim
{
class model_client : public nodelet::Nodelet
{
private:
  ros::ServiceClient client;
  ros::ServiceClient record_client;
  ros::ServiceClient estimate_client;
  ros::ServiceServer server;

  double value_;
  std::vector<double> ground_truth;
  boost::shared_ptr<boost::thread> spinThread_;

public:
  model_client() : value_(0)
  {
	ground_truth.push_back(585.7561);
	ground_truth.push_back(585.7561);
	ground_truth.push_back(320.5);
	ground_truth.push_back(240.5);
  }
  virtual void onInit();
  void spin();
  bool step(franka_cal_sim::RLSrv::Request &req, franka_cal_sim::RLSrv::Response &res);
};

}  // namespace franka_cal_sim

#endif /* INCLUDE_SIMULATION_SERVER_H_ */
