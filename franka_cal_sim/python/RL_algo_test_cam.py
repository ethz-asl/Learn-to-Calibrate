'''
This is for single trajectory test!
Usually we use RL_algo_long_test.py instead.
'''
###
import rospy
from franka_cal_sim.srv import *
import cv2
from cv_bridge import CvBridge
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, GRU, Masking
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import datetime
import os
import tensorflow as tf
import matplotlib.pyplot as plt

import random
from collections import deque
from tempfile import TemporaryFile


def step(state, action):
    rospy.wait_for_service('/model_client/rl_service')
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    take_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
    resp1 = take_step(0, state.tolist(), action.tolist())
    return resp1.next_state, resp1.reward, resp1.done


def reset():
    rospy.wait_for_service('/model_client/rl_service')
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    reset_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
    resp1 = reset_step(1, [], [])
    return resp1.next_state, resp1.reward, resp1.done


def simulation(actions):
    obs_dim = 4
    action_dim = 36
    rospy.init_node('RL_client', anonymous=True)

    num_trials = 10000
    trial_len = actions.shape[0] - 1

    print(actions.shape)

    reset()
    cur_state = np.zeros(obs_dim)
    test_reward_sum = 0
    test_reward_sum = 0
    obs_list = []
    act_list = []
    cur_state = cur_state.reshape((1, obs_dim))
    obs_list.append(cur_state)
    act_list.append(actions[0, :].reshape((1, action_dim)))

    for j in range(trial_len):
        #env.render()
        print("step:" + str(j))

        obs_seq = np.asarray(obs_list)
        print("obs_seq" + str(obs_seq))
        act_seq = np.asarray(act_list)
        obs_seq = obs_seq.reshape((1, -1, obs_dim))
        act_seq = act_seq.reshape((1, -1, action_dim))
        step_num = trial_len - j
        action = actions[j + 1, :]
        action = action.reshape((action_dim))
        cur_state = cur_state.reshape(obs_dim)
        new_state, reward, done = step(cur_state, action)
        test_reward_sum += reward
        rospy.loginfo(rospy.get_caller_id() + 'got reward %s', reward)
        if j == (trial_len - 1):
            done = True

        new_state = np.asarray(new_state).reshape((1, obs_dim))
        action = action.reshape((1, action_dim))

        obs_list.append(new_state)
        act_list.append(action)
        cur_state = new_state
    reset()


if __name__ == '__main__':
    #test
    action_path = rospy.get_param("/rl_client/action_path")
    actions = np.loadtxt(action_path)
    simulation(actions)

    #hand designed action
    actions = np.zeros((5, 36))

    #pitch
    actions[1, 27] = 0.02

    #roll
    actions[2, 2] = 0.02
    actions[2, 13] = 0.02
    actions[2, 19] = -0.02
    #yaw
    actions[3, 2] = -0.02
    actions[3, 7] = 0.02
    actions[3, 31] = 0.02
    #x
    #actions[3,1] = 0.02
    # #y
    # actions[5,7] = 0.02
    # #z
    # actions[3,13] = 0.02
    #8
    actions[4, 7] = 0.02
    actions[4, 15] = 0.02
    actions[4, 21] = 0.02
    actions[4, 31] = -0.02
    actions *= 5.1
    #simulation(actions)

    #random
    # rand_actions = np.random.uniform(low=-0.02*np.ones((5,36)), high=0.02*np.ones((5,36)))
    # python_path = rospy.get_param("/rl_client/python_path")
    # np.savetxt(python_path+"rand_actions.txt",rand_actions)
    # rand_actions = np.loadtxt(python_path+"rand_actions.txt")
    # simulation(rand_actions)
