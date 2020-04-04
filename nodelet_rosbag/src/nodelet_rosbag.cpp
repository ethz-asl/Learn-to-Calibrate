#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>

#include <nodelet_rosbag/RecordAction.h>
#include <nodelet_rosbag/StartAction.h>
#include <nodelet_rosbag/StopAction.h>
#include <nodelet_rosbag/SubscribeAction.h>
#include <nodelet_rosbag/nodelet_rosbag.h>

namespace nodelet_rosbag
{
void NodeRosbagImpl::start_callback()
{
  boost::mutex::scoped_lock(rosbag_mode_mtx_);
  if (recording_)
  {
    for (int i = 0; i < rosbag_record_topics_.size(); ++i)
    {
      // subscribe to the data topic of interest
      ros::Subscriber subscriber =
          private_nh_.subscribe(rosbag_record_topics_[i], 10, &NodeRosbagImpl::record_callback, this);
      record_subscribers_.push_back(subscriber);
    }
  }
  else
  {
    rosbag::View view;
    ros::Time start_time = view.getBeginTime();
    ros::Time end_time = ros::TIME_MAX;
    rosbag::TopicQuery topic_query(rosbag_record_topics_);
    view.addQuery(bag_, topic_query, start_time, end_time);

    BOOST_FOREACH (rosbag::MessageInstance const m, view)
    {
      if (recording_)
      {
        break;
      }

      std::map<std::string, ros::Publisher>::iterator it = playback_publishers_.find(m.getTopic());
      if (it == playback_publishers_.end())
      {
        ros::AdvertiseOptions advertise_opts(m.getTopic(), 10, m.getMD5Sum(), m.getDataType(),
                                             m.getMessageDefinition());

        ros::Publisher publisher = private_nh_.advertise(advertise_opts);

        playback_publishers_.insert(playback_publishers_.begin(),
                                    std::pair<std::string, ros::Publisher>(m.getTopic(), publisher));
      }

      topic_tools::ShapeShifter::ConstPtr s = m.instantiate<topic_tools::ShapeShifter>();
      playback_publishers_[m.getTopic()].publish(s);
    }
  }
}

void NodeRosbagImpl::stop_callback()
{
  boost::mutex::scoped_lock(rosbag_mode_mtx_);
  if (recording_)
  {
    for (int i = 0; i < rosbag_record_topics_.size(); ++i)
    {
      private_nh_.shutdown();
    }
  }
  else
  {
    std::map<std::string, ros::Publisher>::iterator it;
    for (it = playback_publishers_.begin(); it != playback_publishers_.end(); ++it)
    {
      std::pair<std::string, ros::Publisher> pair = *it;
      pair.second.shutdown();
    }
  }
  bag_.close();
}

void NodeRosbagImpl::mode_callback()
{
  boost::mutex::scoped_lock(rosbag_mode_mtx_);
  recording_ = !recording_;
  // TODO(esteve): add error recovery
  bag_.close();
  if (recording_)
  {
    bag_.open(rosbag_path_, rosbag::bagmode::Write);
  }
  else
  {
    for (int i = 0; i < rosbag_record_topics_.size(); ++i)
    {
      private_nh_.shutdown();
    }
    bag_.open(rosbag_path_, rosbag::bagmode::Read);
  }
}

void NodeRosbagImpl::record_callback(const ros::MessageEvent<topic_tools::ShapeShifter const>& event)
{
  boost::mutex::scoped_lock(rosbag_bag_mtx_);
  ros::M_string& header = event.getConnectionHeader();
  topic_tools::ShapeShifter::ConstPtr message = event.getMessage();
  bag_.write(header["topic"], ros::Time::now(), message);
}
}  // namespace nodelet_rosbag
