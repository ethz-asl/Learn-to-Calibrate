#!/usr/bin/env python

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from copy import deepcopy
from gym.spaces import Discrete, Dict, Box
import rospy
from franka_cal_sim.srv import*



class CamCalibrEnv(gym.Env):
    def __init__(self, env_config=None):
        if env_config is None:
            self._init()
        else:
            self._init(**env_config)

    def _init(self, one_card_dealer=False, card_values=None):
        #36*1 parameters
        self.action_space = Box(low=-0.02, high=0.02, shape=(36,), dtype=np.float32)
        #4*1 observations
        self.observation_space = Box(low = -1000, high = 1000, shape = (4,), dtype=np.float32)
        self.seed()
        self.obs = self.reset()
        self.num_steps = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        rospy.wait_for_service('/model_client/rl_service')
        rospy.loginfo(rospy.get_caller_id() + 'begin service')
        reset_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
        resp1 = reset_step(1, [], [])
        return np.ones((4,))*500

    def step(self, action):
        self.num_steps += 1
        info = {'take action': action}
        rospy.wait_for_service('/model_client/rl_service')
        rospy.loginfo(rospy.get_caller_id() + 'begin service')
        take_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
        resp1 = take_step(0, self.obs.tolist(), action.tolist())
        info.update({'get new cal param': resp1.next_state,'get rewards': resp1.reward})
        self.obs = np.asarray(resp1.next_state)
        print(info)
        return self.obs.reshape((4,)), resp1.reward, resp1.done, info

    def render(self):
        pass

    def set_phase(self,phase):
        if phase==0:
            self.action_space = Box(low=-0.03, high=0.03, shape=(1, 36), dtype=np.float32)
        if phase==1:
            self.action_space = Box(low=-0.05, high=0.05, shape=(1, 36), dtype=np.float32)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

class CamCalibrEnv_seq(gym.Env):
    def __init__(self, env_config=None):
        if env_config is None:
            self._init()
        else:
            self._init(**env_config)

    def _init(self):
        #36*1 parameters
        self.action_space = Box(low=-0.05, high=0.05, shape=(36,), dtype=np.float32)
        #4*1 observations
        self.observation_space = Box(low = -1000, high = 1000, shape = (6,40), dtype=np.float32)
        self.seed()
        self.obs_list = []
        self.act_list = []
        self.obs_seq = self.reset()
        self.num_steps = 0
        self.phase = 0
        self.bound = 0.02

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        rospy.wait_for_service('/model_client/rl_service')
        rospy.loginfo(rospy.get_caller_id() + 'begin service')
        reset_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
        resp1 = reset_step(1, [], [])
        #append
        self.obs = np.ones(4)*500
        self.obs_list = []
        self.act_list = []
        self.obs_list.append(normalize(self.obs))
        self.act_list.append(normalize(0.01*np.ones(36)))
        #reshape
        self.obs_seq = np.asarray(self.obs_list)
        self.act_seq = np.asarray(self.act_list)
        self.obs_seq = self.obs_seq.reshape((-1, 4))
        self.act_seq = self.act_seq.reshape((-1, 36))
        #padding
        #padding
        pad_width = 5
        self.obs_seq = np.pad(self.obs_seq,((pad_width,0),(0,0)),'constant')
        self.act_seq = np.pad(self.act_seq,((pad_width,0),(0,0)),'constant')
        return np.concatenate([self.obs_seq,self.act_seq],axis=1)

    def step(self, action):
        info = {'original action': action}
        action = np.clip(action,-self.bound,self.bound)
        self.num_steps += 1
        info.update({'take action': action})
        rospy.wait_for_service('/model_client/rl_service')
        rospy.loginfo(rospy.get_caller_id() + 'begin service')
        take_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
        resp1 = take_step(0, self.obs.tolist(), action.tolist())

        self.obs = np.asarray(resp1.next_state)
        #append
        self.obs_list.append(normalize(self.obs))
        self.act_list.append(normalize(action))
        if len(self.obs_list)>6:
            self.obs_list.pop(0)
            self.act_list.pop(0)
        #reshape
        self.obs_seq = np.asarray(self.obs_list)
        self.act_seq = np.asarray(self.act_list)
        self.obs_seq = self.obs_seq.reshape((-1, 4))
        self.act_seq = self.act_seq.reshape((-1, 36))
        #padding
        #padding
        pad_width = 6-np.size(self.obs_seq,0)
        self.obs_seq = np.pad(self.obs_seq,((pad_width,0),(0,0)),'constant')
        self.act_seq = np.pad(self.act_seq,((pad_width,0),(0,0)),'constant')
        info.update({'get new cal param': resp1.next_state,'get rewards': resp1.reward})
        info.update({'get new seq': self.obs_seq,'phase': self.phase,'act_space_bound':self.bound})
        print(info)
        done = resp1.done
        if self.num_steps==6:
            done=1
        return np.concatenate([self.obs_seq,self.act_seq],axis=1), resp1.reward, done, info

    def render(self):
        pass

    def set_phase(self,phase):
        self.phase = phase
        if phase==0:
            self.bound = 0.02
            self.action_space = Box(low=-0.05, high=0.05, shape=(36,), dtype=np.float32)
        if phase==1:
            self.bound = 0.05
            self.action_space = Box(low=-0.08, high=0.08, shape=(36,), dtype=np.float32)



class imuCalibrEnv_seq(gym.Env):
    def __init__(self, env_config=None):
        if env_config is None:
            self._init()
        else:
            self._init(**env_config)

    def _init(self):
        #36*1 parameters
        self.action_space = Box(low=-2, high=2, shape=(36,), dtype=np.float32)
        #4*1 observations
        self.observation_space = Box(low = -1000, high = 1000, shape = (3,48), dtype=np.float32)
        self.seed()
        self.obs_list = []
        self.act_list = []
        self.obs_seq = self.reset()
        self.num_steps = 0
        self.phase = 0
        self.bound = 2.

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        rospy.wait_for_service('/model_client/rl_service')
        rospy.loginfo(rospy.get_caller_id() + 'begin service')
        reset_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
        resp1 = reset_step(1, [], [])
        #append
        self.obs = np.zeros((12,))*0.0
        self.obs_list = []
        self.act_list = []
        self.obs_list.append(self.obs)
        self.act_list.append(np.ones(36)*0.0)
        #reshape
        self.obs_seq = np.asarray(self.obs_list)
        self.act_seq = np.asarray(self.act_list)
        self.obs_seq = self.obs_seq.reshape((-1, 12))
        self.act_seq = self.act_seq.reshape((-1, 36))
        #padding
        #padding
        pad_width = 2
        self.obs_seq = np.pad(self.obs_seq,((pad_width,0),(0,0)),'constant')
        self.act_seq = np.pad(self.act_seq,((pad_width,0),(0,0)),'constant')
        return np.concatenate([self.obs_seq,self.act_seq],axis=1)

    def step(self, action):
        info = {'original action': action}
        #for ddpg:
        s = 0.00
        action_d = action*0.01
        #action_d = np.clip(action_d,-self.bound,self.bound)
        self.num_steps += 1
        info.update({'take action': action_d})
        rospy.wait_for_service('/model_client/rl_service')
        rospy.loginfo(rospy.get_caller_id() + 'begin service')
        take_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
        resp1 = take_step(0, self.obs.tolist(), action_d.tolist())

        self.obs = np.asarray(resp1.next_state)
        #append
        self.obs_list.append(self.obs)
        #return back
        self.act_list.append(action)
        if len(self.obs_list)>3:
            self.obs_list.pop(0)
            self.act_list.pop(0)
        #reshape
        self.obs_seq = np.asarray(self.obs_list)
        self.act_seq = np.asarray(self.act_list)
        self.obs_seq = self.obs_seq.reshape((-1, 12))
        self.act_seq = self.act_seq.reshape((-1, 36))
        #padding
        #padding
        pad_width = 3-np.size(self.obs_seq,0)
        self.obs_seq = np.pad(self.obs_seq,((pad_width,0),(0,0)),'constant')
        self.act_seq = np.pad(self.act_seq,((pad_width,0),(0,0)),'constant')
        info.update({'get new cal param': resp1.next_state,'get rewards': resp1.reward})
        info.update({'get new seq': self.obs_seq,'phase': self.phase,'act_space_bound':self.bound})
        print(info)
        done = resp1.done
        if self.num_steps==3:
            done=1
        return np.concatenate([self.obs_seq,self.act_seq],axis=1), resp1.reward, done, info

    def render(self):
        pass

    def set_phase(self,phase):
        self.phase = phase
        if phase==0:
            self.bound = 2.
            self.action_space = Box(low=-2., high=2., shape=(36,), dtype=np.float32)
        if phase==1:
            self.bound = 0.05
            self.action_space = Box(low=-0.05, high=0.05, shape=(36,), dtype=np.float32)


class CamImuCalibrEnv_seq(gym.Env):
    def __init__(self, env_config=None):
        if env_config is None:
            self._init()
        else:
            self._init(**env_config)

    def _init(self):
        #36*1 parameters
        self.action_space = Box(low=-0.02, high=0.02, shape=(36,), dtype=np.float32)
        #4*1 observations
        self.observation_space = Box(low = -1000, high = 1000, shape = (3,56), dtype=np.float32)
        self.seed()
        self.obs_list = []
        self.act_list = []
        self.obs_seq = self.reset()
        self.num_steps = 0
        self.phase = 0
        self.bound = 0.02

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        rospy.wait_for_service('/model_client/rl_service')
        rospy.loginfo(rospy.get_caller_id() + 'begin service')
        reset_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
        resp1 = reset_step(1, [], [])
        #append
        self.obs = np.zeros((20,))*0.0
        self.obs_list = []
        self.act_list = []
        self.obs_list.append(self.obs)
        self.act_list.append(0.01*np.ones(36))
        #reshape
        self.obs_seq = np.asarray(self.obs_list)
        self.act_seq = np.asarray(self.act_list)
        self.obs_seq = self.obs_seq.reshape((-1, 20))
        self.act_seq = self.act_seq.reshape((-1, 36))
        #padding
        #padding
        pad_width = 2
        self.obs_seq = np.pad(self.obs_seq,((pad_width,0),(0,0)),'constant')
        self.act_seq = np.pad(self.act_seq,((pad_width,0),(0,0)),'constant')
        return np.concatenate([self.obs_seq,self.act_seq],axis=1)

    def step(self, action):
        info = {'original action': action}
        action = np.clip(action,-self.bound,self.bound)
        self.num_steps += 1
        info.update({'take action': action})
        rospy.wait_for_service('/model_client/rl_service')
        rospy.loginfo(rospy.get_caller_id() + 'begin service')
        take_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
        resp1 = take_step(0, self.obs.tolist(), action.tolist())

        self.obs = np.asarray(resp1.next_state)
        #append
        self.obs_list.append(self.obs)
        #return back
        self.act_list.append(action)
        if len(self.obs_list)>3:
            self.obs_list.pop(0)
            self.act_list.pop(0)
        #reshape
        self.obs_seq = np.asarray(self.obs_list)
        self.act_seq = np.asarray(self.act_list)
        self.obs_seq = self.obs_seq.reshape((-1, 20))
        self.act_seq = self.act_seq.reshape((-1, 36))
        #padding
        #padding
        pad_width = 3-np.size(self.obs_seq,0)
        self.obs_seq = np.pad(self.obs_seq,((pad_width,0),(0,0)),'constant')
        self.act_seq = np.pad(self.act_seq,((pad_width,0),(0,0)),'constant')
        info.update({'get new cal param': resp1.next_state,'get rewards': resp1.reward})
        info.update({'get new seq': self.obs_seq,'phase': self.phase,'act_space_bound':self.bound})
        print(info)
        done = resp1.done
        if self.num_steps==3:
            done=1
        return np.concatenate([self.obs_seq,self.act_seq],axis=1), resp1.reward, done, info

    def render(self):
        pass

    def set_phase(self,phase):
        self.phase = phase
        if phase==0:
            self.bound = 0.02
            self.action_space = Box(low=-0.02, high=0.02, shape=(36,), dtype=np.float32)
        if phase==1:
            self.bound = 0.05
            self.action_space = Box(low=-0.05, high=0.05, shape=(36,), dtype=np.float32)