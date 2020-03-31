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

    des_link_state.link_name = "base_link";
    des_link_state.reference_frame = "world";

    get_link_state_srv.request.link_name = "base_link";
    get_link_state_srv.request.reference_frame = "world";

    double theta=0;
    double dtheta=0.005;
    int T = 1256;
    double k = 0.5;
    double m = 5;
    while(ros::ok()) {
//       //get current state
//       client2.call(get_link_state_srv);
//       cur_link_state = get_link_state_srv.response.link_state;

	   //TODO: design own trajectory

       //left-right trans
        for(int i=0;i<T;i++)
        {
     	   //get current state
 		   client2.call(get_link_state_srv);
 		   cur_link_state = get_link_state_srv.response.link_state;

 		   //trajectory
     	   theta=theta+dtheta;
     	   twist.linear.x = 0.2*cos(theta)*m;
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

 		   ros::Duration(0.001).sleep();
        }

        //p smooth
        for(int i=0;i<T/4;i++)
		{
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = cur_link_state.twist.linear.x-cur_link_state.pose.position.x*k;
		   twist.linear.y = cur_link_state.twist.linear.y-cur_link_state.pose.position.y*k;
		   twist.linear.z = cur_link_state.twist.linear.z-cur_link_state.pose.position.z*k;
		   twist.angular.z = cur_link_state.twist.angular.z-cur_link_state.pose.orientation.z*k;
		   twist.angular.x = cur_link_state.twist.angular.x-cur_link_state.pose.orientation.x*k;
		   twist.angular.y = cur_link_state.twist.angular.y-cur_link_state.pose.orientation.y*k;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.001).sleep();
		}

        // up-down trans
		 for(int i=0;i<T;i++)
		 {
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = 0;
		   twist.linear.y = 0.1*cos(theta)*m;
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

		   ros::Duration(0.001).sleep();
		 }

		 //p smooth
		for(int i=0;i<T/4;i++)
		{
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = cur_link_state.twist.linear.x-cur_link_state.pose.position.x*k;
		   twist.linear.y = cur_link_state.twist.linear.y-cur_link_state.pose.position.y*k;
		   twist.linear.z = cur_link_state.twist.linear.z-cur_link_state.pose.position.z*k;
		   twist.angular.z = cur_link_state.twist.angular.z-cur_link_state.pose.orientation.z*k;
		   twist.angular.x = cur_link_state.twist.angular.x-cur_link_state.pose.orientation.x*k;
		   twist.angular.y = cur_link_state.twist.angular.y-cur_link_state.pose.orientation.y*k;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.001).sleep();
		}


       //forward-backward trans
       for(int i=0;i<T;i++)
       {
    	   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
    	   theta=theta+dtheta;
    	   twist.linear.x = 0;
		   twist.linear.y = 0;
		   twist.linear.z = 0.15*cos(theta)*m;
		   twist.angular.z = 0.0;
		   twist.angular.x = 0.0;
		   twist.angular.y = 0.0;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.001).sleep();
       }


		 //p smooth
		for(int i=0;i<T/4;i++)
		{
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = cur_link_state.twist.linear.x-cur_link_state.pose.position.x*k;
		   twist.linear.y = cur_link_state.twist.linear.y-cur_link_state.pose.position.y*k;
		   twist.linear.z = cur_link_state.twist.linear.z-cur_link_state.pose.position.z*k;
		   twist.angular.z = cur_link_state.twist.angular.z-cur_link_state.pose.orientation.z*k;
		   twist.angular.x = cur_link_state.twist.angular.x-cur_link_state.pose.orientation.x*k;
		   twist.angular.y = cur_link_state.twist.angular.y-cur_link_state.pose.orientation.y*k;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.001).sleep();
		}



       //up-down
       for(int i=0;i<T;i++)
	   {
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = 0;
		   twist.linear.y = 0.2*cos(theta)*m;
		   twist.linear.z = -0.05*sin(theta*2)*1.7*m;
		   twist.angular.z = 0.0;
		   twist.angular.x = -0.15*cos(theta)*1.3*m;
		   twist.angular.y = 0.0;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.001).sleep();
	   }

		 //p smooth
		for(int i=0;i<T/4;i++)
		{
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = cur_link_state.twist.linear.x-cur_link_state.pose.position.x*k;
		   twist.linear.y = cur_link_state.twist.linear.y-cur_link_state.pose.position.y*k;
		   twist.linear.z = cur_link_state.twist.linear.z-cur_link_state.pose.position.z*k;
		   twist.angular.z = cur_link_state.twist.angular.z-cur_link_state.pose.orientation.z*k;
		   twist.angular.x = cur_link_state.twist.angular.x-cur_link_state.pose.orientation.x*k;
		   twist.angular.y = cur_link_state.twist.angular.y-cur_link_state.pose.orientation.y*k;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.001).sleep();
		}


       //left-right
       for(int i=0;i<T;i++)
	   {
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = 0.2*cos(theta)*m;
		   twist.linear.y = 0;
		   twist.linear.z = -0.05*sin(theta*2)*1.7*m;
		   twist.angular.z = 0.0;
		   twist.angular.x = 0.0;
		   twist.angular.y = 0.2*cos(theta)*1.3*m;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.001).sleep();
	   }


		 //p smooth
		for(int i=0;i<T/4;i++)
		{
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = cur_link_state.twist.linear.x-cur_link_state.pose.position.x*k;
		   twist.linear.y = cur_link_state.twist.linear.y-cur_link_state.pose.position.y*k;
		   twist.linear.z = cur_link_state.twist.linear.z-cur_link_state.pose.position.z*k;
		   twist.angular.z = cur_link_state.twist.angular.z-cur_link_state.pose.orientation.z*k;
		   twist.angular.x = cur_link_state.twist.angular.x-cur_link_state.pose.orientation.x*k;
		   twist.angular.y = cur_link_state.twist.angular.y-cur_link_state.pose.orientation.y*k;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.001).sleep();
		}

       //tilt
	   for(int i=0;i<T;i++)
	   {
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = 0.2*cos(theta)*m;
		   twist.linear.y = 0.2*cos(theta)*m;
		   twist.linear.z = -0.05*sin(theta*2)*1.7*m;
		   twist.angular.z = 0.0;
		   twist.angular.x = -0.15*cos(theta)*m;
		   twist.angular.y = 0.15*cos(theta)*m;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.001).sleep();
	   }


		 //p smooth
		for(int i=0;i<T/4;i++)
		{
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = cur_link_state.twist.linear.x-cur_link_state.pose.position.x*k;
		   twist.linear.y = cur_link_state.twist.linear.y-cur_link_state.pose.position.y*k;
		   twist.linear.z = cur_link_state.twist.linear.z-cur_link_state.pose.position.z*k;
		   twist.angular.z = cur_link_state.twist.angular.z-cur_link_state.pose.orientation.z*k;
		   twist.angular.x = cur_link_state.twist.angular.x-cur_link_state.pose.orientation.x*k;
		   twist.angular.y = cur_link_state.twist.angular.y-cur_link_state.pose.orientation.y*k;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.001).sleep();
		}


	   //tilt
	   for(int i=0;i<T;i++)
	   {
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = 0.2*cos(theta)*m;
		   twist.linear.y = -0.2*cos(theta)*m;
		   twist.linear.z = -0.05*sin(theta*2)*1.7*m;
		   twist.angular.z = 0.0;
		   twist.angular.x = 0.15*cos(theta)*m;
		   twist.angular.y = 0.15*cos(theta)*m;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.001).sleep();
	   }


		 //p smooth
		for(int i=0;i<T/4;i++)
		{
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = cur_link_state.twist.linear.x-cur_link_state.pose.position.x*k;
		   twist.linear.y = cur_link_state.twist.linear.y-cur_link_state.pose.position.y*k;
		   twist.linear.z = cur_link_state.twist.linear.z-cur_link_state.pose.position.z*k;
		   twist.angular.z = cur_link_state.twist.angular.z-cur_link_state.pose.orientation.z*k;
		   twist.angular.x = cur_link_state.twist.angular.x-cur_link_state.pose.orientation.x*k;
		   twist.angular.y = cur_link_state.twist.angular.y-cur_link_state.pose.orientation.y*k;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.001).sleep();
		}

       //pitch
       for(int i=0;i<T;i++)
	   {
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = 0.0;
		   twist.linear.y = 0;
		   twist.linear.z = 0;
		   twist.angular.z = 0.7*cos(2*theta)*m;
		   twist.angular.x = 0;
		   twist.angular.y = 0.0;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.001).sleep();
	   }

		 //p smooth
		for(int i=0;i<T/4;i++)
		{
		   //get current state
		   client2.call(get_link_state_srv);
		   cur_link_state = get_link_state_srv.response.link_state;

		   //trajectory
		   theta=theta+dtheta;
		   twist.linear.x = cur_link_state.twist.linear.x-cur_link_state.pose.position.x*k;
		   twist.linear.y = cur_link_state.twist.linear.y-cur_link_state.pose.position.y*k;
		   twist.linear.z = cur_link_state.twist.linear.z-cur_link_state.pose.position.z*k;
		   twist.angular.z = cur_link_state.twist.angular.z-cur_link_state.pose.orientation.z*k;
		   twist.angular.x = cur_link_state.twist.angular.x-cur_link_state.pose.orientation.x*k;
		   twist.angular.y = cur_link_state.twist.angular.y-cur_link_state.pose.orientation.y*k;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.001).sleep();
		}

       //8-shape
       for(int i=0;i<2*T;i++)
       {
    	   //get current state
		  client2.call(get_link_state_srv);
		  cur_link_state = get_link_state_srv.response.link_state;

		  //trajectory
		   theta = theta+dtheta;
		   double x = -0.4*sin(theta+1.5708)*m;
		   double y = 0.4*cos(2*(theta+1.5708))*m;
		   twist.linear.x = x;
		   twist.linear.y = y;
		   twist.linear.z = 0;
		   twist.angular.z = 0.0;
		   twist.angular.x = 0.1*cos(2*theta)*m;
		   twist.angular.y = -0.1*cos(theta)*m;

		   //call the service
		   des_link_state.twist = twist;
		   des_link_state.pose = cur_link_state.pose;
		   set_link_state_srv.request.link_state = des_link_state;
		   client.call(set_link_state_srv);
		   ros::spinOnce();

		   ros::Duration(0.001).sleep();
       }
     }
    return 0;
}
