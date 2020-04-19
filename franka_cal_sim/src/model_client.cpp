#include <pluginlib/class_list_macros.h>
#include "model_client.h"

namespace franka_cal_sim
{
void model_client::onInit()
{
  ros::NodeHandle &n = getPrivateNodeHandle();
  n.getParam("value", value_);
  // define link state client
  client = n.serviceClient<franka_cal_sim::actionSrv>("/simulation_server/internal_service");
  record_client = n.serviceClient<franka_cal_sim::recordSrv>("/record_service");
  estimate_client = n.serviceClient<franka_cal_sim::estimateSrv>("/estimate_service");
  server = n.advertiseService("rl_service", &model_client::step, this);
  //		spinThread_ = boost::shared_ptr< boost::thread >
  //		            (new boost::thread(boost::bind(&model_client::spin, this)));
}

bool model_client::step(franka_cal_sim::RLSrv::Request &req, franka_cal_sim::RLSrv::Response &res)
{
  franka_cal_sim::actionSrv action_srv;
  franka_cal_sim::recordSrv record_srv;
  franka_cal_sim::estimateSrv estimate_srv;

  // call record service
  record_srv.request.start = 1;
  record_client.call(record_srv);

  // call action service
  // if the command is reset
  if (req.reset == 1)
  {
    int act_len = 36;
    action_srv.request.action.clear();
    // random action
    ////need better way
    for (int i = 0; i < act_len; i++)
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
    // return result
    std::vector<double> next_state_;
    double reward_ = 0;
    bool done_ = 1;
    res.next_state = next_state_;
    res.reward = reward_;
    res.done = done_;
  }
  else
  {
    // if command include actions

    // get action from request
    action_srv.request.action = req.action;
    // augment rotation
    for (int i = 18; i < 36; i++)
    {
      action_srv.request.action[i] *= 2.5;
      if (i > 23 && i < 30)
      {
        action_srv.request.action[i] *= 2;
      }
    }
    // pass it to the action service
    client.call(action_srv);

    ROS_INFO("result received %f", action_srv.response.path_len);

    // call estimate service
    estimate_srv.request.reset = 0;
    estimate_srv.request.params = req.state;
    estimate_client.call(estimate_srv);

    // return the result of the model
    // reward: error change, path length, coverage, observability
    res.next_state = estimate_srv.response.par_upd;
    double pre_error = 0;
    for (int i = 0; i < req.state.size(); i++)
    {
      pre_error += pow(req.state[i] - ground_truth[i], 2);
    }
    pre_error = sqrt(pre_error);
    double post_error = 0;
    for (int i = 0; i < res.next_state.size(); i++)
    {
      post_error += pow(res.next_state[i] - ground_truth[i], 2);
    }
    post_error = sqrt(post_error);
    double err_change = post_error - pre_error;

    res.reward = (estimate_srv.response.obs + estimate_srv.response.coverage - err_change / 300 -
                  action_srv.response.path_len / 4) *
                 100;
    ROS_INFO("I got error %f", post_error);
    ROS_INFO("I got params %f", res.next_state);
    ROS_INFO("I got err_change %f", post_error - pre_error);
    ROS_INFO("I got coverage change %f", estimate_srv.response.coverage);
    ROS_INFO("I got obs change %f", estimate_srv.response.obs);
    ROS_INFO("I got path_len %f", action_srv.response.path_len);
    res.done = 0;
    if (post_error < 5)
    {
      res.reward += 500;
      res.done = 1;
    }
  }

  ros::spinOnce();
  return true;
}

PLUGINLIB_EXPORT_CLASS(franka_cal_sim::model_client, nodelet::Nodelet)
}  // namespace franka_cal_sim
