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
    resp1 = take_step(0, state, action.tolist())
    return resp1.next_state, resp1.reward, resp1.done


def reset():
    rospy.wait_for_service('/model_client/rl_service')
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    reset_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
    resp1 = reset_step(1, [], [])
    return resp1.next_state, resp1.reward, resp1.done


def simulation():
    reset()
    state = np.ones(4) * 500
    outfile = TemporaryFile()
    outfile.seek(0)
    action = np.loadtxt(
        '/home/yunke/prl_proj/panda_ws/src/franka_cal_sim/action.txt',
        dtype=float)
    for t in range(10):
        reset()
        state, reward, done = step(state, action)
        rospy.loginfo(rospy.get_caller_id() + 'got reward %s', reward)


if __name__ == '__main__':
    simulation()
