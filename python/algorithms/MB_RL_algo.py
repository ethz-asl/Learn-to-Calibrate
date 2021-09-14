#!/usr/bin/env python


import rospy
from franka_cal_sim_single.srv import*
import cv2
from cv_bridge import CvBridge
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, GRU, Masking
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.compat.v1.keras.backend as K
import datetime
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tempfile import TemporaryFile

import random
from collections import deque
import copy

def stack_samples(samples):
    array = np.array(samples)

    current_act_hists = np.stack(array[:,0]).reshape((array.shape[0],-1,array[0,0].shape[2]))
    actions = np.stack(array[:,1]).reshape((array.shape[0],-1))
    rewards = np.stack(array[:,2]).reshape((array.shape[0],-1))
    new_act_hists = np.stack(array[:,3]).reshape((array.shape[0],-1,array[0,3].shape[2]))
    dones = np.stack(array[:,4]).reshape((array.shape[0],1))

    return current_act_hists, actions, rewards, new_act_hists, dones

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
    def __init__(self):
        # self.sess = sess

        self.trans_learning_rate = 1e-4  #1e-4 #5e-5 #5e-5
        self.reward_learning_rate = 1e-4   #1e-4 #5e-5 #5e-5

        self.num_trial = 1
        self.epsilon = .9
        self.epsilon_decay = .9999 #0.999
        self.gamma = .90
        self.tau   = .01
        self.obs_dim = 6
        self.act_dim = 5
        # ===================================================================== #
        #                               transition Model                             #
        # ===================================================================== #
        self.memory = deque(maxlen=4000)



        # ===================================================================== #
        #                             Reward Model                             #
        # ===================================================================== #

        self.reward_act_hist_input, self.reward_action_input, \
        self.reward_model = self.create_reward_model()

        # Initialize for later gradient calculations
        # self.sess.run(tf.compat.v1.global_variables_initializer())

        #save
        self.checkpoint_path = "training_1/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)

        # self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        # self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # #self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=1)
        # self.checkpoint = tf.train.Checkpoint(optimizer=self.optimize, model=self.actor_model)



    # ========================================================================= #
    #                              Model Definitions                            #


    def create_reward_model(self):
        act_hist_input = Input(shape=(None,self.act_dim))
        mask_action_input = Masking(mask_value=0.)(act_hist_input)
        #simplified network structure
        reward_rnn,state_h2 = GRU(256, return_state=True)([mask_action_input])

        action_input = Input(shape=self.act_dim)
        action_h2    = Dense(256)(action_input)

        merged    = Concatenate()([state_h2, action_h2])
        merged_h1 = Dense(256, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model  = Model([act_hist_input,action_input],output)

        adam  = Adam(lr=self.reward_learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return act_hist_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_act_hist, action, reward, new_act_hist, done):
        self.memory.append([cur_act_hist, action, reward, new_act_hist, done])




    def _train_reward(self, samples):
        cur_act_hists, actions, rewards, new_act_hists, dones = stack_samples(samples)
        evaluation = self.reward_model.fit([cur_act_hists, actions], rewards, batch_size = 32, epochs = 3, verbose=0)
        print(evaluation.history)
        return evaluation.history['loss']

    def train(self):

        #samples = random.sample(self.memory, len(self.memory)/2)
        samples = np.asarray(self.memory)
        self.samples = samples
        reward_loss = self._train_reward(samples)
        return reward_loss


    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def plan(self, cur_act_hist,step_num):
        #given current status, plan a traj in step num using learned model
        tmp_act_hist = copy.copy(cur_act_hist)
        total_reward = 0
        final_action = np.zeros(self.act_dim)
        for step in range(step_num):
            action = np.random.uniform(-0.5,0.5,int(self.act_dim))
            if step==0:
                final_action = copy.copy(action)

            action = action.reshape((1,-1))
            #print("obs_seq"+str(obs_seq))
            act_seq = np.asarray(tmp_act_hist)
            act_seq = act_seq.reshape((1, -1, self.act_dim))
            pred_reward = self.reward_model.predict([act_seq,action])

            tmp_act_hist.append(action)
            total_reward+=pred_reward

        return total_reward, final_action


    def act(self, cur_act_hist, step_num, sample_num, israndom):
        #random action
        if israndom:
            action = np.random.uniform(-0.5,0.5,int(self.act_dim))
            return action

        #MPC action, sample sample_num trajs, pick the one with highest reward, choose that action
        max_reward = -100
        final_action = np.zeros(self.act_dim)
        for i in range(sample_num):
            total_reward, action = self.plan(cur_act_hist,step_num)
            if total_reward>max_reward:
                final_action = copy.copy(action)
                max_reward = total_reward

        return final_action




def normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm+1e-16)

def step(action, last):
    rospy.wait_for_service('/model_client/rl_service')
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    take_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
    resp1 = take_step(0, action.tolist(), last)
    return resp1.reward, resp1.done

def reset():
    rospy.wait_for_service('/model_client/rl_service')
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    reset_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
    resp1 = reset_step(1, [], 0)
    return resp1.reward, resp1.done



def simulation():
    #record the reward trend
    reward_list = []
    test_reward_list = []
    trans_loss_trend = []
    reward_loss_trend = []

    #save path:
    actor_checkpoint = rospy.get_param("/rl_client/actor_checkpoint")
    critic_checkpoint = rospy.get_param("/rl_client/critic_checkpoint")
    fig_path = rospy.get_param("/rl_client/figure_path")
    result_path = rospy.get_param("/rl_client/result_path")
    test_result_path = rospy.get_param("/rl_client/test_result_path")

    #init
    obs_dim = 6
    action_dim = 5
    rospy.init_node('RL_client', anonymous=True)
    # sess = tf.compat.v1.Session()
    # K.set_session(sess)
    actor_critic = ActorCritic()

    num_trials = 10000
    trial_len  = 5

    for i in range(num_trials):

        reset()
        reward_sum = 0
        obs_list = []
        act_list = []
        act_list.append(np.zeros((1,action_dim)))
        last = 0
        #initially chose random actions
        israndom = 0
        if i<5:
            israndom = 1

        for j in range(trial_len):
            #env.render()
            print("trial:" + str(i))
            print("step:" + str(j))
            act_seq = np.asarray(act_list)
            print("act_seq"+str(act_seq))
            act_seq = act_seq.reshape((1, -1, action_dim))
            step_num = trial_len-j
            action = actor_critic.act(act_list,step_num,20,israndom)
            action = action.reshape((action_dim))
            if j == (trial_len - 1):
                last = 1
            reward, done = step(action, last)
            reward_sum += reward
            rospy.loginfo(rospy.get_caller_id() + 'got reward %s',reward)
            if j == (trial_len - 1):
                done = True


            #train the agent
            if j%5==0 and len(actor_critic.memory)>32:
                reward_loss = actor_critic.train()
                reward_loss_trend.append(reward_loss)

            action = action.reshape((1,action_dim))

            act_list.append(action)
            next_act_seq = np.asarray(act_list)
            next_act_seq = next_act_seq.reshape((1, -1, action_dim))

            #padding
            pad_width = trial_len-np.size(act_seq,1)
            print(pad_width)
            act_seq = np.pad(act_seq,((0,0),(pad_width,0),(0,0)),'constant')
            next_act_seq = np.pad(next_act_seq,((0,0),(pad_width,0),(0,0)),'constant')
            #print(obs_seq.shape)
            #print(next_obs_seq.shape)

            actor_critic.remember(act_seq, action, reward, next_act_seq, done)

            if done:
                rospy.loginfo(rospy.get_caller_id() + 'got total reward %s',reward_sum)
                reward_list.append(reward_sum)
                break

        if i % 5 == 0:
            actor_critic.reward_model.save_weights(critic_checkpoint)
            fig, ax = plt.subplots()
            ax.plot(reward_list)
            fig.savefig(fig_path)
            fig_r, ax_r= plt.subplots()
            ax_r.plot(reward_loss_trend)
            fig_r.savefig('/home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/reward_loss.png')

            np.savetxt(result_path, reward_list, fmt='%f')

            #test
            reset()
            test_reward_sum = 0
            test_reward_sum = 0
            act_list = []
            act_list.append(np.zeros((1,action_dim)))
            last = 0
            for j in range(trial_len):
                #env.render()
                print("test:" + str(i))
                print("step:" + str(j))

                act_seq = np.asarray(act_list)
                act_seq = act_seq.reshape((1, -1, action_dim))
                step_num = trial_len-j
                action = actor_critic.act(act_list,step_num,20,0)
                action = action.reshape((action_dim))
                if j == (trial_len - 1):
                    last = 1
                reward, done = step(action,last)
                test_reward_sum += reward
                rospy.loginfo(rospy.get_caller_id() + 'got reward %s',reward)
                if j == (trial_len - 1):
                    done = True


                action = action.reshape((1,action_dim))

                act_list.append(action)
                if done:
                    rospy.loginfo(rospy.get_caller_id() + 'got total reward %s',test_reward_sum)
                    test_reward_list.append(test_reward_sum)
                    fig1, ax1 = plt.subplots()
                    ax1.plot(test_reward_list)
                    fig1.savefig('/home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/test_reward.png')
                    np.savetxt(test_result_path, test_reward_list, fmt='%f')
                    break



if __name__ == '__main__':
    simulation()
