/*
 * action_to_sim.cpp
 *
 *  Created on: Mar 28, 2020
 *      Author: yunke
 */

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
#include <random_numbers/random_numbers.h>
#include <cam_cal_sim/actionSrv.h>

using namespace std;


int main(int argc, char **argv) {
	ros::init(argc, argv, "internal_client");
	ros::NodeHandle n;

	//define link state client
	ros::ServiceClient client = n.serviceClient<cam_cal_sim::actionSrv>("/firefly/internal_service");
    cam_cal_sim::actionSrv action_srv;
    int act_len = 36;
    action_srv.request.act_len = act_len;

	 while(ros::ok()) {
		 action_srv.request.action.clear();
		 for(int i=0;i<act_len;i++)
		 {
			random_numbers::RandomNumberGenerator rand_gen;
			double cur_act = rand_gen.uniformReal(-0.05,0.05);
			if(i>17)
			{
				cur_act *=5;
			}
			action_srv.request.action.push_back(cur_act);
		 }
		 client.call(action_srv);
		 ROS_INFO("result received %f",action_srv.response.path_len);
		 ros::spinOnce();
	 }
    return 0;
}


