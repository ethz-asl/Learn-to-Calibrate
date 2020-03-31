/*
 * simulation_server.h
 *
 *  Created on: Mar 31, 2020
 *      Author: yunke
 */

#ifndef INCLUDE_SIMULATION_SERVER_H_
#define INCLUDE_SIMULATION_SERVER_H_

#include <ros/ros.h>
#include <gazebo_msgs/ModelState.h>
#include <geometry_msgs/Pose.h>
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/SetLinkState.h>
#include <gazebo_msgs/GetLinkState.h>
#include <gazebo_msgs/LinkState.h>
#include <tf/transform_datatypes.h>
#include <tf/LinearMath/Quaternion.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include <string>
#include <cam_cal_sim/actionSrv.h>
#include <std_srvs/Empty.h>
#include <nodelet/nodelet.h>
#include <std_msgs/Float64.h>


namespace cam_cal_sim
{

    class simulation_server : public nodelet::Nodelet
    {
		private:
			ros::ServiceClient client;
			ros::ServiceClient client2;
			ros::ServiceServer service;
			double value_;

		public:
			simulation_server()
			  : value_(0)
			  {}
			virtual void onInit();
			bool model(cam_cal_sim::actionSrv::Request &req, cam_cal_sim::actionSrv::Response &res);
    };

}

#endif /* INCLUDE_SIMULATION_SERVER_H_ */
