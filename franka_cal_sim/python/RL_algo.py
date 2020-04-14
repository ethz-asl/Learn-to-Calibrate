#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that listens to std_msgs/Strings published
## to the 'chatter' topic

import rospy
from franka_cal_sim.srv import *
import numpy as np
import tensorflow as tf
import cv2
from cv_bridge import CvBridge


def step(state, action):
    rospy.wait_for_service('/model_client/rl_service')
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    take_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
    resp1 = take_step(0, state, action)
    return resp1.next_state, resp1.reward, resp1.done


def reset():
    rospy.wait_for_service('/model_client/rl_service')
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    reset_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
    resp1 = reset_step(1, [], [])
    return resp1.next_state, resp1.reward, resp1.done


def simulation():
    rospy.init_node('RL_client', anonymous=True)
    reset()
    state = np.zeros(4)
    for t in range(10):
        action = np.random.rand(36) * 0.1 - 0.05
        state, reward, done = step(state, action)
        rospy.loginfo(rospy.get_caller_id() + 'got reward %s', reward)


if __name__ == '__main__':
    simulation()
