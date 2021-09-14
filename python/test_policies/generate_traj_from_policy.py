'''
This is for single trajectory test!
Usually we use RL_algo_long_test.py instead.
'''
###
import rospy
from franka_cal_sim_single.srv import*
import cv2
from cv_bridge import CvBridge
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt

import random
from collections import deque
from tempfile import TemporaryFile
from rosbag_recorder.srv import RecordTopics, StopRecording



def step(action, last):
    rospy.wait_for_service('/model_client/rl_service')
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    take_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
    
    try:
        resp1 = take_step(0, action.tolist(), last, 0)
    except:
        return np.zeros((18,)), 0, 0
    return resp1.next_state, resp1.reward, resp1.done

def reset(restart):
    rospy.wait_for_service('/model_client/rl_service')
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    reset_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
    try:
        resp1 = reset_step(1, [], 0, restart)
    except:
        return 0
    return resp1.reward


def simulation():
    
    actions = np.loadtxt('generated_trajs/new_test_actions_real_7.txt', dtype=float)
    rospy.wait_for_service('/record_topics')
    reset(0)
    #start recording
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    record = rospy.ServiceProxy('/record_topics', RecordTopics)
    camera_topic = rospy.get_param("/rl_client/camera_topic")
    imu_topic = rospy.get_param("/rl_client/imu_topic")
    resp1 = record('real_exp_data.bag', [camera_topic, imu_topic])
    for i in range(2):
        for t in range(actions.shape[0]):
            #execute the action
            next_state, reward, done = step(actions[t,:], 0)
            rospy.loginfo(rospy.get_caller_id() + 'got reward %s',reward)
    # end recording
    stop_record = rospy.ServiceProxy('/stop_recording', StopRecording)
    resp1 = stop_record('real_exp_data.bag')

if __name__ == '__main__':
    simulation()
