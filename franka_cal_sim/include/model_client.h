
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
#include <franka_cal_sim/actionSrv.h>
#include <franka_cal_sim/recordSrv.h>
#include <franka_cal_sim/estimateSrv.h>
#include <franka_cal_sim/RLSrv.h>
#include <rosbag_recorder/RecordTopics.h>
#include <rosbag_recorder/StopRecording.h>
#include <std_srvs/Empty.h>
#include <nodelet/nodelet.h>
#include <std_msgs/Float64.h>
#include <random_numbers/random_numbers.h>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <vector>
#include <nodelet_rosbag/StartAction.h>
#include <nodelet_rosbag/StopAction.h>
#include <nodelet_rosbag/RecordAction.h>
#include <actionlib/client/simple_action_client.h>
#include <actionlib/client/terminal_state.h>
#include <unistd.h>
#include <sensor_msgs/SetCameraInfo.h>
#include <sensor_msgs/CameraInfo.h>
#include <gazebo_msgs/SetJointProperties.h>
#include <gazebo_msgs/ODEJointProperties.h>
#include <random>
#include <chrono>
#include <fstream>
#include <string>



namespace franka_cal_sim
{

    class model_client : public nodelet::Nodelet
    {
		private:
		//server of the RL agent, client of simulation, recording and estimation
			ros::ServiceClient client;
			ros::ServiceClient record_client;
			ros::ServiceClient estimate_client;
			ros::ServiceServer server;
            ros::ServiceClient bag_client;
            ros::ServiceClient stop_client;
            ros::ServiceClient camera_client;
            ros::ServiceClient extrinsic_client;
            ros::Subscriber camera_sub;

			double value_;
			double length;
			//ground truth of calibration parameters
			std::vector<double> ground_truth;
            std::vector<double> cal_params;
			//calibrate intrinsics or extrinsics
			bool if_intrinsic;
			bool if_change_sensors;
			//path of kalibr result
            std::string kalibr_path;
			boost::shared_ptr<boost::thread> spinThread_;

		public:
			model_client()
			  : value_(0)
			  {
			    ground_truth.push_back(585.7561);
			    ground_truth.push_back(585.7561);
			    ground_truth.push_back(320.5);
			    ground_truth.push_back(240.5);
			    cal_params.push_back(0);
			    cal_params.push_back(0);
			    cal_params.push_back(0);
			    cal_params.push_back(0);
			    length = 0;
			  }
			virtual void onInit();
			void spin();
			//service for RL client
			bool step(franka_cal_sim::RLSrv::Request &req,franka_cal_sim::RLSrv::Response &res);
			//change ground truth
			void change_settings();
			//change kalibr inputs
			void change_cam_yaml(double fx, double fy, double cx, double cy);

    };

}

#endif /* INCLUDE_SIMULATION_SERVER_H_ */
