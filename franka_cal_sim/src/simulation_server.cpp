/*
 * action_to_sim.cpp
 *
 *  Created on: Mar 28, 2020
 *      Author: yunke
 */

#include "simulation_server.h"
#include <pluginlib/class_list_macros.h>

// watch the capitalization carefully


namespace franka_cal_sim
{
    void simulation_server::onInit()
    {
        NODELET_DEBUG("Initializing nodelet simulation server");
        ros::NodeHandle &n=getPrivateNodeHandle();
        n.getParam("value", value_);

		//define the server
		service = n.advertiseService("internal_service",&simulation_server::model,this);
		ROS_INFO("Successfully launched node.");
    }




	bool simulation_server::model(franka_cal_sim::actionSrv::Request &req,franka_cal_sim::actionSrv::Response &res)
	{
		//current and desire link states
		//moveit group setup
		static const std::string PLANNING_GROUP = "panda_arm";
		moveit::planning_interface::MoveGroupInterface move_group(PLANNING_GROUP);
		moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
		geometry_msgs::Pose final_pose;
		geometry_msgs::Quaternion quat;
		tf::Quaternion quat_tf;
		//define initial position, should have large controbility
		final_pose.position.x = 0.4;
		final_pose.position.y = 0.3;
		final_pose.position.z = 0.47;
		double R = 1.57;
		double P = 0.8;
		double Y = 1.57;
		quat_tf.setRPY(R,P,Y);
		tf::quaternionTFToMsg(quat_tf,quat);
		final_pose.orientation = quat;

        //waypoints
		std::vector<geometry_msgs::Pose> waypoints;
//		if(init)
//		{
//			geometry_msgs::PoseStamped cur_pose = move_group.getCurrentPose(move_group.getEndEffector());
//            waypoints.push_back(cur_pose.pose);
//            init = 0;
//		}
//        waypoints.push_back(final_pose);
		geometry_msgs::Pose pose;


		//read the action array:[-0.5,0.5], cos,cos2,sin,sin2,cos4,sin4: 6*6 = 36 elements
		int len  = req.action.size();
		double* action = new double[len];
		for(int i=0;i<len;i++)
		{
			action[i] = req.action[i];
		}

		//begin simulate the trajectory
		double dtheta = 0.1;
		double pi = 3.14159265358979;
		int num_of_step = int(2*pi/dtheta);
		double theta = 0;
		double path_len = 0;
		std_srvs::Empty srv;
		double R_ = 1.5708;
		double P_ = 0.8;
		double Y_ = 1.5708;
		for(int i=0;i<num_of_step;i++)
		{
		   //get current state

		   //trajectory
		   theta=theta+dtheta;
		   pose.position.x = action[0]*cos(theta)+action[1]*sin(theta)+action[2]*cos(2*theta)+action[3]*sin(2*theta);
		   pose.position.x += action[4]*cos(4*theta)+action[5]*sin(4*theta)-(action[0]+action[2]+action[4])+0.4;
		   pose.position.y = action[6]*cos(theta)+action[7]*sin(theta)+action[8]*cos(2*theta)+action[9]*sin(2*theta);
		   pose.position.y += action[10]*cos(4*theta)+action[11]*sin(4*theta)-(action[6]+action[8]+action[10])+0.3;
		   pose.position.z = action[12]*cos(theta)+action[13]*sin(theta)+action[14]*cos(2*theta)+action[15]*sin(2*theta);
		   pose.position.z += action[16]*cos(4*theta)+action[17]*sin(4*theta)-(action[12]+action[14]+action[16])+0.47;
		   R = action[18]*cos(theta)+action[19]*sin(theta)+action[20]*cos(2*theta)+action[21]*sin(2*theta);
		   R += action[22]*cos(4*theta)+action[23]*sin(4*theta)-(action[18]+action[20]+action[22])+1.5708;
		   P = action[24]*cos(theta)+action[25]*sin(theta)+action[26]*cos(2*theta)+action[27]*sin(2*theta);
		   P += action[28]*cos(4*theta)+action[29]*sin(4*theta)-(action[24]+action[26]+action[28])+0.8;
		   Y = action[30]*cos(theta)+action[31]*sin(theta)+action[32]*cos(2*theta)+action[33]*sin(2*theta);
		   Y += action[34]*cos(4*theta)+action[35]*sin(4*theta)-(action[30]+action[32]+action[34])+1.5708;

		   quat_tf.setRPY(R,P,Y);
		   tf::quaternionTFToMsg(quat_tf,quat);
		   pose.orientation = quat;

		   //add length
		   if(i>0)
           {
               double cur_len = 0;
               cur_len += pow(pose.position.x-waypoints[i-1].position.x,2);
               cur_len += pow(pose.position.y-waypoints[i-1].position.y,2);
               cur_len += pow(pose.position.z-waypoints[i-1].position.z,2);
               cur_len = sqrt(cur_len);
               cur_len += sqrt(pow(R-R_,2)+pow(P-P_,2)+pow(Y-Y_,2));
               R_ = R;
               Y_ = Y;
               P_ = P;
               path_len += cur_len;
           }


		   //record way point
           waypoints.push_back(pose);
		   //ros::spinOnce();

		}
		//final pose
		waypoints.push_back(final_pose);

		//set constraints
		moveit_msgs::OrientationConstraint ocm;
		ocm.link_name = "panda_link7";
		ocm.header.frame_id = "panda_link0";
		ocm.absolute_x_axis_tolerance = 0.3;
	    ocm.absolute_y_axis_tolerance = 0.3;
		ocm.absolute_z_axis_tolerance = 0.3;
		ocm.weight = 1;
		moveit_msgs::Constraints test_constraints;
		test_constraints.orientation_constraints.push_back(ocm);
		move_group.setPathConstraints(test_constraints);
		//plan
		move_group.setMaxVelocityScalingFactor(0.1);
		moveit_msgs::RobotTrajectory trajectory;
	    const double jump_threshold = 100;
		const double eef_step = 0.005;
		double fraction = move_group.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);

		bool unpaused = ros::service::call("/gazebo/unpause_physics", srv);
		moveit::planning_interface::MoveGroupInterface::Plan my_plan;
		my_plan.trajectory_ = trajectory;
		move_group.execute(my_plan);
        ros::Duration(5).sleep();
		bool paused = ros::service::call("/gazebo/pause_physics", srv);
		//get the observability and new params from estimator

		res.path_len = path_len;


		return true;
	}

	PLUGINLIB_EXPORT_CLASS(franka_cal_sim::simulation_server, nodelet::Nodelet)
}
