
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
#include <tf2/LinearMath/Quaternion.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include <string>
#include <franka_cal_sim_single/actionSrv.h>
#include <franka_cal_sim_single/recordSrv.h>
#include <franka_cal_sim_single/estimateSrv.h>
#include <franka_cal_sim_single/RLSrv.h>
#include <franka_cal_sim_single/distortSrv.h>
#include <franka_cal_sim_single/getBoardCenterSrv.h>

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
#include <tf/tf.h>
#include <iostream>

namespace franka_cal_sim_single
{

    class model_client : public nodelet::Nodelet
    {
		private:
			ros::NodeHandle* n;
			ros::ServiceClient client;
			ros::ServiceClient record_client;
			ros::ServiceClient estimate_client;
			ros::ServiceServer server;
            ros::ServiceClient bag_client;
            ros::ServiceClient stop_client;
			ros::ServiceClient distort_reset_client;
			ros::ServiceClient get_board_center_client;

			double value_;
			boost::shared_ptr<boost::thread> spinThread_;
			double path_len;
			double total_translation;
			double total_rotation;
			int max_time_steps;
			std::vector<double> ground_truth;
			std::vector<double> cur_pose;
			double step_path_len;
			double step_translation;
			double step_rotation;

			bool if_intrinsic;
			bool if_ext_policy;
			std::vector<double> max_difference;
			std::vector<double> cur_difference;
			std::vector<double> pose_range;

			std::string camera_topic;
			std::string imu_topic;
			bool real_test;

		public:
			model_client()
			  : value_(0)
			  {
			    for(int i=0;i<6;i++)
                {
			        cur_pose.push_back(0);
			        max_difference.push_back(0);
					cur_difference.push_back(0);
                }
			    path_len = 0;
				//add boundary
				pose_range.push_back(0.08);
				pose_range.push_back(0.1);//0.1 //0.12
				pose_range.push_back(0.08);
				pose_range.push_back(0.17);//0.17
				pose_range.push_back(0.15);//0.12 //0.15
				pose_range.push_back(0.12);//0.12
			  }
			virtual void onInit();
			void spin();
			bool step(franka_cal_sim_single::RLSrv::Request &req,franka_cal_sim_single::RLSrv::Response &res);
			void compute_next_pose(std::vector<double> action, bool last);
    };

}

#endif /* INCLUDE_SIMULATION_SERVER_H_ */
