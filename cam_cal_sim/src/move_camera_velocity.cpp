//uses ROS service to move checkerboard in Gazebo
//turn off gravity, so object does not fall after repositioning
//can tune desired range of displacements and rotations

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
#include <iostream>
#include <string>
using namespace std; 


int main(int argc, char **argv) {
    ros::init(argc, argv, "move_gazebo_model");
    ros::NodeHandle n;

    //define link state client
    ros::ServiceClient client = n.serviceClient<gazebo_msgs::SetLinkState>("/gazebo/set_link_state");
    ros::ServiceClient client2 = n.serviceClient<gazebo_msgs::GetLinkState>("/gazebo/get_link_state");
    gazebo_msgs::GetLinkState get_link_state_srv;
    gazebo_msgs::SetLinkState set_link_state_srv;
    gazebo_msgs::LinkState des_link_state;
    gazebo_msgs::LinkState cur_link_state;
    geometry_msgs::Twist twist;

    //desired poses
    geometry_msgs::Pose pose;
    geometry_msgs::Quaternion quat;

    pose.position.x = 0;
   	pose.position.y = 0;
   	pose.position.z = 0.5;

    des_link_state.link_name = "camera_base";
    des_link_state.reference_frame = "world";

    get_link_state_srv.request.link_name = "camera_base";
    get_link_state_srv.request.reference_frame = "world";

    double theta=0;
    double dtheta=0.02;
    while(ros::ok()) {
//       //get current state
//       client2.call(get_link_state_srv);
//       cur_link_state = get_link_state_srv.response.link_state;

	   //TODO: design own trajectory

       //left-right trans
        for(int i=0;i<325;i++)
        {
     	   //get current state
 		   client2.call(get_link_state_srv);
 		   cur_link_state = get_link_state_srv.response.link_state;

 		   //trajectory
     	   theta=theta+dtheta;
     	   twist.linear.x = 0.07*cos(theta)*2;
 		   twist.linear.y = 0;
 		   twist.linear.z = 0;
 		   twist.angular.z = 0.0;
 		   twist.angular.x = 0.0;
 		   twist.angular.y = 0.0;

 		   //call the service
 		   des_link_state.twist = twist;
 		   des_link_state.pose = cur_link_state.pose;
 		   set_link_state_srv.request.link_state = des_link_state;
 		   client.call(set_link_state_srv);
 		   ros::spinOnce();

 		   ros::Duration(0.01).sleep();
        }

        // up-down trans
		 for(int i=0;i<314;i++)
		 {
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = 0;
		   twist.linear.y = 0.05*cos(theta)*2;
		   twist.linear.z = 0;
		   twist.angular.z = 0.0;
		   twist.angular.x = 0.0;
		   twist.angular.y = 0.0;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.01).sleep();
		 }

       //forward-backward trans
       for(int i=0;i<314;i++)
       {
    	   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
    	   theta=theta+dtheta;
    	   twist.linear.x = 0;
		   twist.linear.y = 0;
		   twist.linear.z = 0.08*cos(theta)*2;
		   twist.angular.z = 0.0;
		   twist.angular.x = 0.0;
		   twist.angular.y = 0.0;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.01).sleep();
       }

       //up-down
       for(int i=0;i<314;i++)
	   {
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = 0;
		   twist.linear.y = 0.1*cos(theta)*2;
		   twist.linear.z = -0.05*sin(theta*2)*1.7*2;
		   twist.angular.z = 0.0;
		   twist.angular.x = -0.35*cos(theta)*1.3*2;
		   twist.angular.y = 0;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.01).sleep();
	   }

       //left-right
       for(int i=0;i<314;i++)
	   {
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = 0.1*cos(theta)*2;
		   twist.linear.y = 0;
		   twist.linear.z = -0.05*sin(theta*2)*1.7*2;
		   twist.angular.z = 0.0;
		   twist.angular.x = 0;
		   twist.angular.y = 0.35*cos(theta)*1.3*2;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.01).sleep();
	   }

       //tilt
	   for(int i=0;i<314;i++)
	   {
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = 0.1*cos(theta)*2;
		   twist.linear.y = 0.1*cos(theta)*2;
		   twist.linear.z = -0.05*sin(theta*2)*1.7*2;
		   twist.angular.z = 0.0;
		   twist.angular.x = -0.35*cos(theta)*2;
		   twist.angular.y = 0.35*cos(theta)*2;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.01).sleep();
	   }

	   //tilt
	   for(int i=0;i<314;i++)
	   {
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = 0.1*cos(theta)*2;
		   twist.linear.y = -0.1*cos(theta)*2;
		   twist.linear.z = -0.05*sin(theta*2)*1.7*2;
		   twist.angular.z = 0.0;
		   twist.angular.x = 0.35*cos(theta)*2;
		   twist.angular.y = 0.35*cos(theta)*2;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.01).sleep();
	   }

       //pitch
       for(int i=0;i<314;i++)
	   {
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = 0.0;
		   twist.linear.y = 0;
		   twist.linear.z = 0;
		   twist.angular.z = 0.5*cos(2*theta)*2;
		   twist.angular.x = 0;
		   twist.angular.y = 0.0;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.01).sleep();
	   }

       //8-shape
       for(int i=0;i<628;i++)
       {
    	   //get current state
		  client2.call(get_link_state_srv);
		  cur_link_state = get_link_state_srv.response.link_state;

		  //trajectory
		   theta = theta+dtheta;
		   double x = -0.1*sin(theta+1.5708)*2;
		   double y = 0.1*cos(2*(theta+1.5708))*2;
		   twist.linear.x = x;
		   twist.linear.y = y;
		   twist.linear.z = 0;
		   twist.angular.z = 0.0;
		   twist.angular.x = 0.1*cos(2*theta)*2;
		   twist.angular.y = -0.15*cos(theta)*2;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.01).sleep();
       }
     }
    return 0;
}
