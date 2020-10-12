


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
        record_client = n.serviceClient<franka_cal_sim::recordSrv>("/record_service");
        estimate_client = n.serviceClient<franka_cal_sim::estimateSrv>("/estimate_service");
        bag_client = n.serviceClient<rosbag_recorder::RecordTopics>("/record_topics");
        stop_client = n.serviceClient<rosbag_recorder::StopRecording>("/stop_recording");
        server = n.advertiseService("rl_service",&model_client::step,this);
        camera_client = n.serviceClient<sensor_msgs::SetCameraInfo>("/simple_camera/set_camera_info");
        extrinsic_client = n.serviceClient<gazebo_msgs::SetJointProperties>("/gazebo/set_joint_properties");

        n.getParam("/rl_client/if_calibrate_intrinsic",if_intrinsic);
        n.getParam("/rl_client/if_change_sensors",if_change_sensors);
        if(if_intrinsic)
        {
            n.getParam("/rl_client/cam_ground_truth",ground_truth);
        }
        else
        {
            n.getParam("/rl_client/imu_cam_ground_truth",ground_truth);
        }
        n.getParam("/rl_client/kalibr_path",kalibr_path);
	}

	void model_client::change_cam_yaml(double fx, double fy, double cx, double cy)
    {
        std::deque <std::string> text;
        std::string s;

        // load the file
        std::string filename = kalibr_path+"camchain.yaml";
        std::ifstream inf( filename );  // You must supply the proper filename
        while (getline( inf, s ))
        {
            if(s.at(2)=='i')
            {
                s = "  intrinsics: [";
                s+=std::to_string(fx);
                s+=", ";
                s+=std::to_string(fy);
                s+=", ";
                s+=std::to_string(cx);
                s+=", ";
                s+=std::to_string(cy);
                s+="]";
            }
            text.push_back( s );
        }

        inf.close();


        // rewrite the new text to file
        std::ofstream outf( filename );

        for (std::deque <std::string> ::iterator
                     line  = text.begin();
             line != text.end();
             ++line)
            outf << *line << '\n';
        outf.close();

    }

	void model_client::change_settings()
    {
	    double w = 640,h = 480;
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::normal_distribution<double> extr_dist(0.03,0.01);

        double new_finger = extr_dist(generator);
        //ROS_INFO(set_cam_srv.response.status_message);
        //ROS_INFO("new finger %f",new_finger);

//        //imu extrinsic
//        gazebo_msgs::SetJointProperties set_joint_srv;
//        gazebo_msgs::ODEJointProperties ode_joint_prop;
//        //left
//        set_joint_srv.request.joint_name = "panda_finger_joint1";
//        ode_joint_prop.hiStop.push_back(new_finger);
//        ode_joint_prop.loStop.push_back(new_finger);
//        set_joint_srv.request.ode_joint_config = ode_joint_prop;
//        extrinsic_client.call(set_joint_srv);
//        //right
//        set_joint_srv.request.joint_name = "panda_finger_joint2";
//        set_joint_srv.request.ode_joint_config = ode_joint_prop;
//        extrinsic_client.call(set_joint_srv);

        //change ground truth camera intrinsic value
        if(if_intrinsic){
            ROS_INFO("\n\n\n");
            ros::param::get("/rl_client/cam_ground_truth",ground_truth);
            ROS_INFO("ground_truth %f",ground_truth[0]);
        }
        else
        {
//            ground_truth.clear();
//            ground_truth.push_back(2*new_finger);
//            ground_truth.push_back(0);
//            ground_truth.push_back(-0.1);
//            ground_truth.push_back(0);
//            ground_truth.push_back(0);
//            ground_truth.push_back(1.5708);
            ROS_INFO("\n\n\n");
            ros::param::get("/rl_client/imu_cam_ground_truth",ground_truth);
            ROS_INFO("ground_truth %f",ground_truth[0]);


        }

    }

	bool model_client::step(franka_cal_sim::RLSrv::Request &req,franka_cal_sim::RLSrv::Response &res)
	{

        franka_cal_sim::actionSrv action_srv;
        franka_cal_sim::recordSrv record_srv;
        franka_cal_sim::estimateSrv estimate_srv;

         //call record service
         record_srv.request.start=1;
         record_client.call(record_srv);
         //call record bag
         rosbag_recorder::RecordTopics bag_record_srv;
         rosbag_recorder::StopRecording stop_srv;
         bag_record_srv.request.name = "data_tmp.bag";
         stop_srv.request.name = "data_tmp.bag";

         bag_record_srv.request.topics.push_back("/simple_camera/image_raw");
         bag_record_srv.request.topics.push_back("/imu_real");


         //call action service
         //if the command is reset
         if(req.reset==1)
         {
             //reset sensor properties
             if(if_change_sensors)
             {
                 change_settings();
             }


             action_srv.request.reset = 1;
             int act_len = 36;
             action_srv.request.action.clear();
             //random action
             ////need better way
             for(int i=0;i<act_len;i++)
             {
                 random_numbers::RandomNumberGenerator rand_gen;
                 double cur_act = 0.02;
//                 if(i>17)
//                 {
//                     cur_act *=1;
//                 }
//				if(i>23&&i<30)
//				{
//					cur_act *=2;
//				}
                 action_srv.request.action.push_back(cur_act);
             }


             client.call(action_srv);
             std::vector<double> state_;
             estimate_srv.request.params = state_;
             estimate_srv.request.reset = 1;
             estimate_client.call(estimate_srv);
             //return result
             std::vector<double> next_state_;
             double reward_ = 0;
             bool done_ = 1;
             res.next_state = next_state_;
             res.reward = reward_;
             res.done = done_;
             cal_params.clear();
             for(int i=0;i<ground_truth.size();i++)
             {
                 cal_params.push_back(0);
             }
             length = 0;
         }
         else
         {
             action_srv.request.reset = 0;
             //if command include actions
//             //start bag
             bag_client.call(bag_record_srv);
             for(int j=0;j<20000;j++);
             //get action from request
             action_srv.request.action = req.action;
             //augment rotation
             for(int i=18;i<36;i++)
             {
                 action_srv.request.action[i]*=2.5;
                 if(i>23&&i<30)
				{
                    action_srv.request.action[i]*=2;
				}
             }
             //pass it to the action service

             client.call(action_srv);

             //stop bag
             //ros::service::call("/stop_recording",stop_srv);
             //ros::Duration(2).sleep();

             ROS_INFO("result received %f",action_srv.response.path_len);
             ROS_INFO("req state %f",req.state.size());
             stop_client.call(stop_srv);
             if(stop_srv.response.success ==1)
             {
                 ROS_INFO("Stop successfully");
             }


             //call estimate service
             estimate_srv.request.reset=0;
             estimate_srv.request.params=req.state;
             estimate_client.call(estimate_srv);

             //return the result of the model
             //reward: error change, path length, coverage, observability
             res.next_state = estimate_srv.response.par_upd;
             double pre_error = 0;
             for(int i=0;i<ground_truth.size();i++)
             {
                 pre_error+=pow(cal_params[i]-ground_truth[i],2);
             }
             pre_error = sqrt(pre_error);
             double post_error = 0;
             for(int i=0;i<ground_truth.size();i++)
             {
                 post_error+=pow(res.next_state[i]-ground_truth[i],2);
             }
             post_error = sqrt(post_error);
             double err_change = post_error-pre_error;
             cal_params.clear();
             for(int i=0;i<ground_truth.size();i++)
             {
                 cal_params.push_back(estimate_srv.response.par_upd[i]);
             }
             res.next_state.clear();
             for(int i=ground_truth.size();i<estimate_srv.response.par_upd.size();i++)
             {
                 res.next_state.push_back(estimate_srv.response.par_upd[i]);
             }
             double norm;
             for(int i=0;i<ground_truth.size();i++)
             {
                 norm+=ground_truth[i]*ground_truth[i];
             }
             norm = sqrt(norm);
             /////change err_change with new
             if(if_intrinsic)//err_change/300
             {
                 ROS_INFO("I got ground_truth %f %f %f %f %f %f",ground_truth[0],ground_truth[1],ground_truth[2],ground_truth[3],ground_truth[4],ground_truth[5]);
                 res.reward = (estimate_srv.response.obs+estimate_srv.response.coverage-action_srv.response.path_len/5-err_change/norm*3);
                 //reproj_error
                 //res.reward = (estimate_srv.response.obs+estimate_srv.response.coverage-action_srv.response.path_len/5);
             }

             else {
                 res.reward = (estimate_srv.response.obs + estimate_srv.response.coverage - err_change / pre_error -
                               action_srv.response.path_len);
                 //use reproj err
                 res.reward = (estimate_srv.response.obs + estimate_srv.response.coverage -
                               action_srv.response.path_len);
             }

             length+=action_srv.response.path_len;
             ROS_INFO("I got error %f",post_error);
             ROS_INFO("I got next state %f",res.next_state);

             ROS_INFO("I got err_change %f",post_error-pre_error);
             ROS_INFO("I got coverage change %f",estimate_srv.response.coverage);
             ROS_INFO("I got obs change %f",estimate_srv.response.obs);
             ROS_INFO("I got path_len %f",action_srv.response.path_len);
             ROS_INFO("I got reward %f",res.reward);
             ROS_INFO("I got total length %f",length);
             res.done = 0;
//             if(post_error<0.02&&!if_intrinsic)
//             {
//                 res.reward+=5;
//                 //res.done = 1;
//             }
//             if(post_error<5&&if_intrinsic)
//             {
//                 res.reward+=5;
//                 //res.done = 1;
//             }


         }

         ros::spinOnce();
         return true;

	}

	PLUGINLIB_EXPORT_CLASS(franka_cal_sim::model_client, nodelet::Nodelet)
}
