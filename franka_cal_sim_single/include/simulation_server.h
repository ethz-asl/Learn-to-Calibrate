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
#include <vector>
#include <franka_cal_sim_single/actionSrv.h>
#include <std_srvs/Empty.h>
#include <nodelet/nodelet.h>
#include <std_msgs/Float64.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <franka_cal_sim_single/getBoardCenterSrv.h>

#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>

#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>

#include <moveit_visual_tools/moveit_visual_tools.h>
#include <Eigen/Dense>


namespace franka_cal_sim_single
{

    class simulation_server : public nodelet::Nodelet
    {
		private:
			ros::ServiceClient client;
			ros::ServiceClient client2;
			ros::ServiceServer service;

			double value_;
			bool init;
			bool if_curve;
			bool real_test;
			bool if_change_sensors;
			geometry_msgs::Pose init_pose;
			geometry_msgs::Pose cur_pose;
			std::vector<double> cur_req_pose;
			//direction of update position
			bool tracking_flag;
			double last_center_x;
			double last_center_y;

		public:
			simulation_server()
			  : value_(0),init(1),if_curve(0)
			  {
				init_pose.position.x = 0.43; //0.5
				init_pose.position.y = 0.0; //0.35
				init_pose.position.z = 0.45; //0.47
				cur_pose.position.x = 0.43; //0.5
				cur_pose.position.y = 0.0; //0.35
				cur_pose.position.z = 0.45; //0.47
				double R = 3.14; //1.57
				double P = 0.0; //0.8
				double Y = -0.785; //1.57
				tf::Quaternion quat_tf;
				geometry_msgs::Quaternion quat;
				quat_tf.setRPY(R,P,Y);
				tf::quaternionTFToMsg(quat_tf,quat);
				cur_pose.orientation = quat;
				tracking_flag = 0;
				last_center_x = 1.0;
				last_center_y = 1.0;
				
				for(int i=0;i<6;i++)
				{
					cur_req_pose.push_back(0);
				}
			  }
			virtual void onInit();
			bool model(franka_cal_sim_single::actionSrv::Request &req, franka_cal_sim_single::actionSrv::Response &res);
    };

}

#endif /* INCLUDE_SIMULATION_SERVER_H_ */
