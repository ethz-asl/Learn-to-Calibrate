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
#include <random_numbers/random_numbers.h>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>

namespace cam_cal_sim
{

    class model_client : public nodelet::Nodelet
    {
		private:
			ros::ServiceClient client;
			double value_;
			boost::shared_ptr<boost::thread> spinThread_;

		public:
			model_client()
			  : value_(0)
			  {}
			virtual void onInit();
			void spin();
    };

}

#endif /* INCLUDE_SIMULATION_SERVER_H_ */
