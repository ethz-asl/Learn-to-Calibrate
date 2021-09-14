


#include "model_client.h"
#include <pluginlib/class_list_macros.h>


namespace franka_cal_sim_single
{
	void model_client::onInit()
	{
		ros::NodeHandle &n_ = getPrivateNodeHandle();
        n = &n_;
		n->getParam("value", value_);
		//define link state client
		client = n->serviceClient<franka_cal_sim_single::actionSrv>("/simulation_server/internal_service");
        record_client = n->serviceClient<franka_cal_sim_single::recordSrv>("/record_service");
        estimate_client = n->serviceClient<franka_cal_sim_single::estimateSrv>("/estimate_service");
        bag_client = n->serviceClient<rosbag_recorder::RecordTopics>("/record_topics");
        stop_client = n->serviceClient<rosbag_recorder::StopRecording>("/stop_recording");
        distort_reset_client = n->serviceClient<franka_cal_sim_single::distortSrv>("/reset_service");
        server = n->advertiseService("rl_service",&model_client::step,this);
        get_board_center_client = n->serviceClient<franka_cal_sim_single::getBoardCenterSrv>("/get_board_center_service");
        //select ground truth
        n->getParam("/rl_client/imu_cam_ground_truth",ground_truth);
        n->getParam("/rl_client/num_steps",max_time_steps);
        n->getParam("/rl_client/if_intrinsic_policy", if_intrinsic);
        n->getParam("/rl_client/if_ext_policy", if_ext_policy);
        n->getParam("/rl_client/camera_topic", camera_topic);
        n->getParam("/rl_client/imu_topic", imu_topic);
        n->getParam("/rl_client/real_test", real_test);
	}


    void model_client::compute_next_pose(std::vector<double> action, bool last)
    {
        /*
        state:
        [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,-0.08(z),-0.1 (x),-0.08,-0.3,-0.15,-0.15] ~ [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,0.08(z),0.1 (x),0.08(y),0.3,0.15,0.15]
        action:
        [-0.2,-1.57,-1.57,-0.2,-0.2,-0.2] ~ [0.2,1.57,1.57,0.2,0.2,0.2]
        */
        //compute the next pose
        std::vector<double> next_pose;
        ROS_INFO("start compute");
        if(last)
        {
            step_path_len = 0;
            
            for(int i=0;i<6;i++)
            {
                next_pose.push_back(0);
            }

            step_path_len = 0;
            step_translation = 0;
            step_rotation = 0;

            // //compute step path length
            // double angle_distance = 0;
            // for(int i=0;i<6;i++)
            // {
            //     if(i<3) 
            //     {
            //         step_path_len += pow(next_pose[i]-cur_pose[i],2);
            //     }
            //     else
            //     {
            //         if(i==3) step_path_len = sqrt(step_path_len);
            //         angle_distance += pow(next_pose[i]-cur_pose[i],2)*0.2;
            //     }
            // }
            // angle_distance = sqrt(angle_distance);
            // step_path_len+=angle_distance;


            for(int i=0;i<6;i++)
            {
                cur_difference[i] = abs(next_pose[i]-cur_pose[i])/2/pose_range[i];
                cur_pose[i] = next_pose[i];
            }


        }
        else
        {
            //first add position
            double rho = action[0];
            double dx = rho*sin(action[1])*cos(action[2]);
            double dy = rho*sin(action[1])*sin(action[2]);
            double dz = rho*cos(action[1]);
            
            
            next_pose.push_back(cur_pose[0]+dx);
            next_pose.push_back(cur_pose[1]+dy);
            next_pose.push_back(cur_pose[2]+dz);

            //add some motion constraints
            double norm = 0;
            for(int i=0; i<3; i++)
            {
                double bound = pose_range[i];
                next_pose[i] = next_pose[i]>bound?bound:next_pose[i];
                next_pose[i] = next_pose[i]<-bound?-bound:next_pose[i];
            }
            

            for(int i=3;i<6;i++)
            {
                next_pose.push_back(0);
            }

            //compute rotation (action: r,p,y)
            tf::Quaternion q_act,q_cur_pose,q_next_pose;

            q_cur_pose.setRPY(cur_pose[3],cur_pose[4],cur_pose[5]);
            q_act.setRPY(action[5],action[4],action[3]);

            //apply rotation
            q_next_pose = q_act*q_cur_pose;
            tf::Matrix3x3 m(q_next_pose);
            m.getRPY(next_pose[3],next_pose[4],next_pose[5]);

            for(int i=3; i<6; i++)
            {
                //add motion constraints
                double bound = pose_range[i];
                next_pose[i] = next_pose[i]>bound?bound:next_pose[i];
                next_pose[i] = next_pose[i]<-bound?-bound:next_pose[i];
            }

            step_path_len = 0;
            step_translation = 0;
            step_rotation = 0;
            //compute step path length
            double angle_distance = 0;
            for(int i=0;i<6;i++)
            {
                if(i<3) 
                {
                    step_path_len += pow(next_pose[i]-cur_pose[i],2);
                    step_translation += pow(next_pose[i]-cur_pose[i],2);
                }
                else
                {
                    if(i==3) 
                    {
                        step_path_len = sqrt(step_path_len);
                        step_translation = sqrt(step_translation);
                    }
                    angle_distance += pow(next_pose[i]-cur_pose[i],2)*0.2;
                }
            }
            angle_distance = sqrt(angle_distance);
            step_rotation += angle_distance;
            step_path_len += angle_distance;

            //assign next to cur
            for(int i=0;i<6;i++)
            {
                cur_difference[i] = abs(next_pose[i]-cur_pose[i])/pose_range[i]/2;
                cur_pose[i] = next_pose[i];
            }
        }
    }

	bool model_client::step(franka_cal_sim_single::RLSrv::Request &req,franka_cal_sim_single::RLSrv::Response &res)
	{

        franka_cal_sim_single::actionSrv action_srv;
        franka_cal_sim_single::recordSrv record_srv;
        franka_cal_sim_single::estimateSrv estimate_srv;
        franka_cal_sim_single::getBoardCenterSrv get_board_center_srv;

         //call record service
         record_srv.request.start=1;
         record_client.call(record_srv);
         //call record bag
         rosbag_recorder::RecordTopics bag_record_srv;
         rosbag_recorder::StopRecording stop_srv;
         bag_record_srv.request.name = "data_tmp.bag";
         stop_srv.request.name = "data_tmp.bag";

         

         bag_record_srv.request.topics.push_back(camera_topic);
         bag_record_srv.request.topics.push_back(imu_topic);


         //call action service
         //if the command is reset
         if(req.reset==1)
         {
             //reset pose, length
             path_len = 0;
             total_translation = 0;
             total_rotation = 0;
             //cur_pose.clear();
             for(int i=0;i<6;i++)
             {
                 cur_pose.pop_back();
             }
             for(int i=0;i<6;i++)
             {
                 cur_pose.push_back(0);
                 max_difference[i] = 0;
				 cur_difference[i] = 0;
             }
             //send simulation srv to reset pose
             int act_len = 6;
             action_srv.request.pose.clear();
             for(int i=0;i<act_len;i++)
             {
                 action_srv.request.pose.push_back(0);
             }
             action_srv.request.reset = 1;
             action_srv.request.align = 0;

             //loop for aligning image center and board center
             double diff = 1;
             client.call(action_srv);
             do
             {
                client.call(action_srv);
                estimate_srv.request.reset = 1;
                estimate_srv.request.restart = req.restart;
                estimate_client.call(estimate_srv);
                get_board_center_client.call(get_board_center_srv);
                diff = abs(get_board_center_srv.response.center_x-0.5)+abs(get_board_center_srv.response.center_y-0.5);
                action_srv.request.align = 1;
             }while(diff>0.02);
             


             
             //return rlsrv result
             double reward_ = 0;
             bool done_ = 0;
             res.reward = reward_;
             ROS_INFO("reset complete");

             //call reset distortion service
            //  franka_cal_sim_single::distortSrv distort_srv;
            //  distort_srv.request.reset = 1;
            //  distort_reset_client.call(distort_srv);
         }
         else
         {

             compute_next_pose(req.action,req.last);
             ROS_INFO("end compute");
             std::cout<<req.action.size()<<std::endl;
             std::cout<<cur_pose[5]<<std::endl;


             //call record bag service
             if(!if_intrinsic)
                bag_client.call(bag_record_srv);

             ROS_INFO("start simulation");

             //get action from request
             for(int i=0;i<6;i++)
             {
                 action_srv.request.pose.push_back(cur_pose[i]);
             }

             //call simulation service
             action_srv.request.reset = 0;
             client.call(action_srv);

             //call stop record service
             if(!if_intrinsic)
             {
                 stop_client.call(stop_srv);
                if(stop_srv.response.success ==1)
                {
                    ROS_INFO("Stop successfully");
                }
             }
             
             if(real_test)
             {
                 res.reward = 0;
                 res.next_state = cur_pose;
                 res.done = 0;
                 return true;
             }

             res.reward = 0;
             if(!if_intrinsic) //extrinsic calibration get state before call estimation
             {
                 res.done = 1;
                 //compute coverage
                 for(int i=0;i<6;i++)
                 {
                    res.reward += (cur_difference[i] - max_difference[i])>0?100*(cur_difference[i] - max_difference[i]):0;
                    //update max coverage
                    max_difference[i] = (cur_difference[i] - max_difference[i])>0?cur_difference[i]:max_difference[i];
                    //compute done
                    res.done = res.done && (max_difference[i] > 0.3);
                 }
                 //update state
                 res.next_state = max_difference;
             }

             //call estimate service
             estimate_srv.request.reset=0;
             if(if_intrinsic)
                estimate_srv.request.done=0;
             else
                estimate_srv.request.done=res.done;
            estimate_srv.response.cal_err = 1;
            estimate_client.call(estimate_srv);

             
             // intrinsic get state
             if(if_intrinsic)
             {
                 //compute new parameter from estimator
                res.next_state = estimate_srv.response.par_upd;
                //compute done from estimator
                res.done = estimate_srv.response.done;
             }
             else if(!if_ext_policy)
             {
                 //add state
                 for(int i=0; i<estimate_srv.response.par_upd.size(); i++)
                 {
                     res.next_state.push_back(estimate_srv.response.par_upd[i]);
                 }
                 //done is both done 
                 res.done = estimate_srv.response.done && res.done;
             }
             
             //add the pose to state
             for(int i=0; i<6; i++)
             {
                 res.next_state.push_back(cur_pose[i]);
             }

             
             //compute additional reward
             res.reward += (estimate_srv.response.info_gain + estimate_srv.response.empirical - step_path_len*5);

             //reward: error change
             if(res.done)
             {
                 res.reward += 100 + 0.1/estimate_srv.response.cal_err;
             }
                

            //update parameter logs
            //n->setParam("/rl_client/calibration_err",estimate_srv.response.cal_err);
            //updata total path length
            path_len+=step_path_len;
            total_translation += step_translation;
            total_rotation += step_rotation;

             n->setParam("/rl_client/translation_and_rotation",path_len);
             n->setParam("/rl_client/total_translation",total_translation);
             n->setParam("/rl_client/total_rotation",total_rotation);

             ROS_INFO("I got error %f",estimate_srv.response.cal_err);
             ROS_INFO("I got empirical change %f",estimate_srv.response.empirical);
             ROS_INFO("I got info gain change %f",estimate_srv.response.info_gain);
             ROS_INFO("I got current path_len %f",path_len);
             ROS_INFO("I got total translation %f",total_translation);
            ROS_INFO("I got total rotation %f",total_rotation);

             

         }

         ros::spinOnce();
         return true;

	}

	PLUGINLIB_EXPORT_CLASS(franka_cal_sim_single::model_client, nodelet::Nodelet)
}
