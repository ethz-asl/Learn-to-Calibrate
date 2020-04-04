/*
 * action_to_sim.cpp
 *
 *  Created on: Mar 28, 2020
 *      Author: yunke
 */


#include "model_client.h"
#include <pluginlib/class_list_macros.h>

namespace franka_cal_sim
{
	void model_client::onInit()
	{
		ros::NodeHandle &n = getPrivateNodeHandle();
		n.getParam("value", value_);
		//define link state client
		client = n.serviceClient<franka_cal_sim::actionSrv>("/simulation_server/internal_service");

		spinThread_ = boost::shared_ptr< boost::thread >
		            (new boost::thread(boost::bind(&model_client::spin, this)));

	}

	void model_client::spin()
	{
		while(ros::ok()) {
			franka_cal_sim::actionSrv action_srv;
			int act_len = 36;
			action_srv.request.act_len = act_len;
			 action_srv.request.action.clear();
			 for(int i=0;i<act_len;i++)
			 {
				random_numbers::RandomNumberGenerator rand_gen;
				double cur_act = rand_gen.uniformReal(-0.05,0.05);
				if(i>17)
				{
					cur_act *=3;
				}
				if(i>23&&i<30)
				{
					cur_act *=2;
				}
				action_srv.request.action.push_back(cur_act);
			 }
			 client.call(action_srv);
			 ROS_INFO("result received %f",action_srv.response.path_len);
			 ros::spinOnce();
		 }
	}

	PLUGINLIB_EXPORT_CLASS(franka_cal_sim::model_client, nodelet::Nodelet)
}
