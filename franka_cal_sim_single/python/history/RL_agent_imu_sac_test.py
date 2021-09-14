#!/usr/bin/env python


import numpy as np
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.agents.sac import SACTrainer
from ray.tune import run, sample_from
import random
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf

tf = try_import_tf()
from calibration_env import CamCalibrEnv, CamCalibrEnv_seq, imuCalibrEnv_seq, CamImuCalibrEnv_seq
import ray

env = imuCalibrEnv_seq()

state = env.reset()
actions = np.loadtxt('/home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/action_tiny2.txt', dtype=float)
rewards = 0
for i in range(3):
    action = actions[i]
    state,reward,done,_ = env.step(action)
    rewards+=reward

print(rewards)

#hand designed action
actions = np.zeros((7,36))

#pitch
actions[0,27] = 0.02
#roll
actions[1,2] = -0.02
actions[1,13] = 0.02
actions[1,19] = 0.02
#yaw
actions[2,2] = -0.02
actions[2,7] = 0.02
actions[2,31] = 0.02
#x
actions[3,1] = 0.02
#y
actions[4,7] = 0.02
#z
actions[5,13] = 0.02
#8
actions[6,7] = 0.02
actions[6,15] = 0.02
actions[6,21] = 0.02
actions[6,31] = -0.02
actions*=3
rewards = 0
state = env.reset()
for i in range(7):
    action = actions[i]
    state,reward,done,_ = env.step(action)
    rewards+=reward

print(rewards)


#random action
state = env.reset()
actions = np.random.rand(3,36)*0.04-0.02
rewards = 0
for i in range(3):
    action = actions[i]
    state,reward,done,_ = env.step(action)
    rewards+=reward

print(rewards)