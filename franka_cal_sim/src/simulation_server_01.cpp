#include <pluginlib/class_list_macros.h>
#include "simulation_server.h"

// watch the capitalization carefully

namespace franka_cal_sim
{
void simulation_server::onInit()
{
  NODELET_DEBUG("Initializing nodelet simulation server");
  ros::NodeHandle &n = getPrivateNodeHandle();
  n.getParam("value", value_);

  // define link state client
  client = n.serviceClient<gazebo_msgs::SetLinkState>("/gazebo/set_link_state");
  client2 = n.serviceClient<gazebo_msgs::GetLinkState>("/gazebo/get_link_state");
  // define the server
  service = n.advertiseService("internal_service", &simulation_server::model, this);
  ROS_INFO("Successfully launched node.");
}

bool simulation_server::model(franka_cal_sim::actionSrv::Request &req, franka_cal_sim::actionSrv::Response &res)
{
  // current and desire link states
  gazebo_msgs::GetLinkState get_link_state_srv;
  gazebo_msgs::SetLinkState set_link_state_srv;
  gazebo_msgs::LinkState des_link_state;
  gazebo_msgs::LinkState cur_link_state;

  des_link_state.link_name = "camera_base";
  des_link_state.reference_frame = "world";

  get_link_state_srv.request.link_name = "camera_base";
  get_link_state_srv.request.reference_frame = "world";

  geometry_msgs::Twist twist;
  geometry_msgs::Pose pose;

  // read the action array:[-0.5,0.5], cos,cos2,sin,sin2,cos4,sin4: 6*6 = 36 elements
  int len = req.act_len;
  double *action = new double[len];
  for (int i = 0; i < len; i++)
  {
	action[i] = req.action[i];
  }

  // begin simulate the trajectory
  double sim_time = 0.005;
  double dtheta = 0.05;
  double pi = 3.14159265358979;
  int num_of_step = int(2 * pi / dtheta);
  double theta = 0;
  double path_len = 0;
  std_srvs::Empty srv;
  bool unpaused = ros::service::call("/gazebo/unpause_physics", srv);
  for (int i = 0; i < num_of_step; i++)
  {
	// get current state
	client2.call(get_link_state_srv);
	cur_link_state = get_link_state_srv.response.link_state;

	// trajectory
	theta = theta + dtheta;
	pose.position.x =
		action[0] * cos(theta) + action[1] * sin(theta) + action[2] * cos(2 * theta) + action[3] * sin(2 * theta);
	pose.position.x += action[4] * cos(4 * theta) + action[5] * sin(4 * theta) - (action[0] + action[2] + action[4]);
	pose.position.y =
		action[6] * cos(theta) + action[7] * sin(theta) + action[8] * cos(2 * theta) + action[9] * sin(2 * theta);
	pose.position.y += action[10] * cos(4 * theta) + action[11] * sin(4 * theta) - (action[6] + action[8] + action[10]);
	pose.position.z =
		action[12] * cos(theta) + action[13] * sin(theta) + action[14] * cos(2 * theta) + action[15] * sin(2 * theta);
	pose.position.z +=
		action[16] * cos(4 * theta) + action[17] * sin(4 * theta) - (action[12] + action[14] + action[16]);
	twist.angular.z =
		action[18] * cos(theta) + action[19] * sin(theta) + action[20] * cos(2 * theta) + action[21] * sin(2 * theta);
	twist.angular.z += action[22] * cos(4 * theta) + action[23] * sin(4 * theta);
	twist.angular.x =
		action[24] * cos(theta) + action[25] * sin(theta) + action[26] * cos(2 * theta) + action[27] * sin(2 * theta);
	twist.angular.x += action[28] * cos(4 * theta) + action[29] * sin(4 * theta);
	twist.angular.y =
		action[30] * cos(theta) + action[31] * sin(theta) + action[32] * cos(2 * theta) + action[33] * sin(2 * theta);
	twist.angular.y += action[34] * cos(4 * theta) + action[35] * sin(4 * theta);

	// add length
	double cur_len = 0;
	cur_len += pow(pose.position.x - cur_link_state.pose.position.x, 2);
	cur_len += pow(pose.position.y - cur_link_state.pose.position.y, 2);
	cur_len += pow(pose.position.z - cur_link_state.pose.position.z, 2);
	cur_len = sqrt(cur_len);
	cur_len += sqrt(pow(twist.angular.z, 2) + pow(twist.angular.y, 2) + pow(twist.angular.x, 2));
	path_len += cur_len;

	// call the service
	des_link_state.twist = twist;
	des_link_state.pose.position = pose.position;
	des_link_state.pose.orientation = cur_link_state.pose.orientation;
	set_link_state_srv.request.link_state = des_link_state;
	client.call(set_link_state_srv);
	// ros::spinOnce();

	ros::Duration(sim_time).sleep();
  }

  // final state
  gazebo_msgs::LinkState final_link_state;
  final_link_state.link_name = "camera_base";
  final_link_state.reference_frame = "world";
  set_link_state_srv.request.link_state = final_link_state;
  client.call(set_link_state_srv);
  // ros::spinOnce();
  ros::Duration(sim_time).sleep();
  bool paused = ros::service::call("/gazebo/pause_physics", srv);
  // get the observability and new params from estimator
  // call another service
  double obs = 0;
  double dx[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  int dx_len = 10;
  double coverage = 0;

  // send back the response
  res.obs = obs;
  for (int i = 0; i < dx_len; i++)
  {
	res.par_upd.push_back(dx[i]);
  }
  res.coverage = coverage;
  res.path_len = path_len;

  return true;
}

PLUGINLIB_EXPORT_CLASS(franka_cal_sim::simulation_server, nodelet::Nodelet)
}  // namespace franka_cal_sim
