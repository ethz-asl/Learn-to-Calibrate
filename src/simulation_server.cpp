
#include "simulation_server.h"
#include <pluginlib/class_list_macros.h>

// watch the capitalization carefully


namespace franka_cal_sim_single
{
    void simulation_server::onInit()
    {
        NODELET_DEBUG("Initializing nodelet simulation server");
        ros::NodeHandle &n=getPrivateNodeHandle();
        n.getParam("value", value_);

		//define the server
		service = n.advertiseService("internal_service",&simulation_server::model,this);
		client = n.serviceClient<franka_cal_sim_single::getBoardCenterSrv>("/get_board_center_service");
		ROS_INFO("Successfully launched node.");

		//get if extend traj
		n.getParam("/rl_client/if_curve", if_curve);
		n.getParam("/rl_client/real_test", real_test);
		n.getParam("/rl_client/if_change_sensors", if_change_sensors);
    }
    




	bool simulation_server::model(franka_cal_sim_single::actionSrv::Request &req,franka_cal_sim_single::actionSrv::Response &res)
	{
		//current and desire link states
		//moveit group setup
		//set start state of the robot
        std_srvs::Empty srv;
		if(req.reset==1) {
            bool unpaused = ros::service::call("/gazebo/unpause_physics", srv);
        }

		if(req.reset==1 && req.align==0)
		{
			tracking_flag = 0;
			last_center_x = 1.0;
			init_pose.position.y = 0.0; //0.35
			init_pose.position.z = 0.45; //0.47
		}

		static const std::string PLANNING_GROUP = "panda_arm";
		moveit::planning_interface::MoveGroupInterface move_group(PLANNING_GROUP);
		moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
		geometry_msgs::Quaternion quat;
		tf::Quaternion quat_tf;

		//define initial position, should have large controbility
		

		//adjust init pose according to board center
		if(req.reset==1 && req.align==1) {
		franka_cal_sim_single::getBoardCenterSrv get_board_center_srv;
		client.call(get_board_center_srv);
		if(!tracking_flag)
		{
			init_pose.position.y += 0.3*(0.5-get_board_center_srv.response.center_x);
			init_pose.position.z += 0.3*(0.5-get_board_center_srv.response.center_y);
		}
		else
		{
			init_pose.position.y -= 0.3*(0.5-get_board_center_srv.response.center_x);
			init_pose.position.z -= 0.3*(0.5-get_board_center_srv.response.center_y);
		}
		if(abs(get_board_center_srv.response.center_x-0.5)>abs(last_center_x-0.5)-0.001 && abs(init_pose.position.y)>0.1)
			tracking_flag = !tracking_flag;

		ROS_INFO("I got init_y %f",init_pose.position.y);
		ROS_INFO("I got init_z %f",init_pose.position.z);

		last_center_x = get_board_center_srv.response.center_x;
		
		}

		double R = 3.14; //1.57
		double P = 0.0; //0.8
		double Y = -0.785; //1.57
		quat_tf.setRPY(R,P,Y);
		tf::quaternionTFToMsg(quat_tf,quat);
		init_pose.orientation = quat;


        //get next pose
		geometry_msgs::Pose pose;
        pose.position.x = req.pose[0]+init_pose.position.x;
        pose.position.y = req.pose[1]+init_pose.position.y;
        pose.position.z = req.pose[2]+init_pose.position.z;
        tf::Quaternion quat_tf_pose;
        quat_tf_pose.setRPY(req.pose[3],req.pose[4],req.pose[5]);
		
		quat_tf_pose = quat_tf_pose*quat_tf;

		// tf::Quaternion quat_tf_add_rot;
		// quat_tf_add_rot.setRPY(0,0,-0.8);

		//quat_tf_pose = quat_tf_add_rot*quat_tf_pose;
        tf::quaternionTFToMsg(quat_tf_pose,pose.orientation);



		//plan
		if(!if_curve || req.reset==1)
		{
			std::vector<geometry_msgs::Pose> waypoints;
			waypoints.push_back(pose);
			moveit_msgs::RobotTrajectory trajectory;
			std_srvs::Empty srv;
			
			move_group.setPoseTarget(pose);
			
			// double fraction = move_group.computeCartesianPath(waypoints,
			// 											0.001,  // eef_step
			// 											0.0,   // jump_threshold
			// 											trajectory);
			
			if(req.reset!=1 && !real_test) {
            	bool unpaused = ros::service::call("/gazebo/unpause_physics", srv);
        	}
			moveit::planning_interface::MoveGroupInterface::Plan my_plan;
			//my_plan.trajectory_ = trajectory;
			bool success = (move_group.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
			
			move_group.execute(my_plan);

			if(req.reset==1 && if_change_sensors && req.align==0)
			{
				ros::Duration(15).sleep();
			}
			if(!real_test)
				bool paused = ros::service::call("/gazebo/pause_physics", srv);
		}
		

		//extend trajectory
		if(if_curve && req.reset!=1)
		{
			geometry_msgs::Pose pose1, pose2;
			std::vector<geometry_msgs::Pose> waypoints;
			//decide direction of curve
			Eigen::Vector3d cur_to_next(pose.position.x-cur_pose.position.x, pose.position.y-cur_pose.position.y, pose.position.z-cur_pose.position.z);
			Eigen::Vector3d a;
			if(abs(cur_to_next[0])>abs(cur_to_next[1]) && abs(cur_to_next[0])>abs(cur_to_next[2]))
			{
				a<<1,0,0;
				
			}
			else if(abs(cur_to_next[1])>abs(cur_to_next[0]) && abs(cur_to_next[1])>abs(cur_to_next[2]))
			{
				a<<0,1,0;
			}
			else
			{
				a<<0,0,1;
			}
			
			double cos_t = a.dot(cur_to_next)/(a.norm()*cur_to_next.norm());
			Eigen::Vector3d ext_drct = a - cos_t*cur_to_next/cur_to_next.norm();

			//compute waypoints position
			Eigen::Vector3d cur_pos(cur_pose.position.x,cur_pose.position.y,cur_pose.position.z);
			Eigen::Vector3d pos(pose.position.x,pose.position.y,pose.position.z);
			double norm = cur_to_next.norm();
			// Eigen::Vector3d pos1 = cur_pos + (pos - cur_pos)*1/4 + ext_drct*norm*1/10;
			// Eigen::Vector3d pos2 = cur_pos + (pos - cur_pos)*3/4 - ext_drct*norm*1/10;
			int k = 50;

			for(int i=1; i<k; i++)
			{
				Eigen::Vector3d pos1 = cur_pos + (pos - cur_pos)*i/k;
				pose1.position.x = pos1[0]; pose1.position.y = pos1[1]; pose1.position.z = pos1[2];
				quat_tf_pose.setRPY(req.pose[3]*i/k+cur_req_pose[3]*(k-i)/k,req.pose[4]*i/k+cur_req_pose[4]*(k-i)/k,req.pose[5]*i/k+cur_req_pose[5]*(k-i)/k);
				quat_tf_pose = quat_tf_pose*quat_tf;
				tf::quaternionTFToMsg(quat_tf_pose,pose1.orientation);
				waypoints.push_back(pose1);
			}
			
			
			//compute waypoints orientation

			
			
			//execute
			//waypoints.push_back(pose1);
			//waypoints.push_back(pose);
			
			//waypoints.push_back(cur_pose);
			waypoints.push_back(pose);
			// waypoints.push_back(init_pose);
			// waypoints.push_back(cur_pose);
			// waypoints.push_back(pose1);
			// waypoints.push_back(pose2);
			// waypoints.push_back(pose);
			// //waypoints.push_back(pose1);
			// waypoints.push_back(cur_pose);
			// //waypoints.push_back(pose2);
			// waypoints.push_back(pose);
			
			moveit_msgs::RobotTrajectory trajectory;
			double fraction = move_group.computeCartesianPath(waypoints,
														0.02,  // eef_step
														0.0,   // jump_threshold
														trajectory);
			std_srvs::Empty srv;
			if(req.reset!=1 && !real_test) {
            	bool unpaused = ros::service::call("/gazebo/unpause_physics", srv);
        	}
			
			moveit::planning_interface::MoveGroupInterface::Plan my_plan;
			my_plan.trajectory_ = trajectory;
			//bool success = (move_group.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
			move_group.execute(my_plan);
			//ros::Duration(5).sleep();
			// if(req.reset==1 && if_change_sensors)
			// {
			// 	ros::Duration(14).sleep();
			// }
			if(!real_test)
			bool paused = ros::service::call("/gazebo/pause_physics", srv);
		}

		//assign cur_pose
		cur_pose = pose;
		cur_req_pose = req.pose;

        res.success = true;

		return true;
	}

	PLUGINLIB_EXPORT_CLASS(franka_cal_sim_single::simulation_server, nodelet::Nodelet)
}
