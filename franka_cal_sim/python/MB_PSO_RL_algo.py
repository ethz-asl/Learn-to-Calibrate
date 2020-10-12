#!/usr/bin/env python


import rospy
from franka_cal_sim.srv import*
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
import pyswarms as ps

def stack_samples(samples):
    array = np.array(samples)

    current_states = np.stack(array[:,0]).reshape((array.shape[0],-1,array[0,0].shape[2]))
    current_act_hists = np.stack(array[:,1]).reshape((array.shape[0],-1,array[0,1].shape[2]))
    actions = np.stack(array[:,2]).reshape((array.shape[0],-1))
    rewards = np.stack(array[:,3]).reshape((array.shape[0],-1))
    new_states = np.stack(array[:,4]).reshape((array.shape[0],-1,array[0,4].shape[2]))
    new_act_hists = np.stack(array[:,5]).reshape((array.shape[0],-1,array[0,5].shape[2]))
    dones = np.stack(array[:,6]).reshape((array.shape[0],1))

    return current_states, current_act_hists, actions, rewards, new_states, new_act_hists, dones

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
        self.obs_dim = 12
        self.act_dim = 36
        # ===================================================================== #
        #                               transition Model                             #
        # ===================================================================== #
        self.memory = deque(maxlen=4000)


        self.trans_state_input, self.trans_act_hist_input, self.trans_action_input, \
        self.trans_model = self.create_transition_model()

        # ===================================================================== #
        #                             Reward Model                             #
        # ===================================================================== #

        self.reward_state_input, self.reward_act_hist_input, self.reward_action_input, \
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

    def create_transition_model(self):
        # ========================================================================= #
        state_input = Input(shape=(None,self.obs_dim))
        act_hist_input = Input(shape=(None,self.act_dim))
        mask_state_input = Masking(mask_value=0.)(state_input)
        mask_action_input = Masking(mask_value=0.)(act_hist_input)
        #simplified network structure
        trans_rnn,state_h2 = GRU(256, return_state=True)(Concatenate()([mask_state_input,mask_action_input]))

        action_input = Input(shape=self.act_dim)
        action_h2    = Dense(256)(action_input)

        merged    = Concatenate()([state_h2, action_h2])
        merged_h1 = Dense(256, activation='relu')(merged)
        output = Dense(self.obs_dim, activation='linear')(merged_h1)
        model  = Model([state_input,act_hist_input,action_input],output)

        adam  = Adam(lr=self.trans_learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, act_hist_input, action_input, model

    def create_reward_model(self):
        state_input = Input(shape=(None,self.obs_dim))
        act_hist_input = Input(shape=(None,self.act_dim))
        mask_state_input = Masking(mask_value=0.)(state_input)
        mask_action_input = Masking(mask_value=0.)(act_hist_input)
        #simplified network structure
        reward_rnn,state_h2 = GRU(256, return_state=True)(Concatenate()([mask_state_input,mask_action_input]))

        action_input = Input(shape=self.act_dim)
        action_h2    = Dense(256)(action_input)

        merged    = Concatenate()([state_h2, action_h2])
        merged_h1 = Dense(256, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model  = Model([state_input,act_hist_input,action_input],output)

        adam  = Adam(lr=self.reward_learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, act_hist_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, cur_act_hist, action, reward, new_state, new_act_hist, done):
        self.memory.append([cur_state, cur_act_hist, action, reward, new_state, new_act_hist, done])

    def _train_trans(self, samples):

        cur_states, cur_act_hists, actions, rewards, new_states,new_act_hists, _ =  stack_samples(samples)
        new_obs = new_states[:,-1,0:self.obs_dim]
        evaluation = self.trans_model.fit([cur_states,cur_act_hists,actions],new_obs, batch_size = 32, epochs = 3, verbose=0)
        print(evaluation.history)
        return evaluation.history['loss']



    def _train_reward(self, samples):
        cur_states, cur_act_hists, actions, rewards, new_states,new_act_hists, dones = stack_samples(samples)
        evaluation = self.reward_model.fit([cur_states, cur_act_hists, actions], rewards, batch_size = 32, epochs = 3, verbose=0)
        print(evaluation.history)
        return evaluation.history['loss']

    def train(self):

        #samples = random.sample(self.memory, len(self.memory)/2)
        samples = np.asarray(self.memory)
        self.samples = samples
        trans_loss = self._train_trans(samples)
        reward_loss = self._train_reward(samples)
        return trans_loss,reward_loss


# ========================================================================= #
#                              Model Predictions                            #
# ========================================================================= #
actor_critic = ActorCritic()
obs_list = []
act_list = []
global_opt_1 = deque(maxlen=5)
global_opt_2 = deque(maxlen=5)
global_opt_3 = deque(maxlen=5)

def reward_funtion(x):
    #input: next actions shape (step_num*act_dim,1)
    #output: total reward
    global obs_list
    global act_list
    global actor_critic
    step_num = int(x.shape[0]/actor_critic.act_dim)
    tmp_obs_hist = copy.copy(obs_list)
    tmp_act_hist = copy.copy(act_list)
    total_reward = 0
    for step in range(step_num):
        action = x[step*actor_critic.act_dim:(step+1)*actor_critic.act_dim]

        action = action.reshape((1,-1))
        obs_seq = np.asarray(tmp_obs_hist)
        #print("obs_seq"+str(obs_seq))
        act_seq = np.asarray(tmp_act_hist)
        obs_seq = obs_seq.reshape((1, -1, actor_critic.obs_dim))
        act_seq = act_seq.reshape((1, -1, actor_critic.act_dim))
        pred_obs = actor_critic.trans_model.predict([obs_seq,act_seq,action])
        pred_reward = actor_critic.reward_model.predict([obs_seq,act_seq,action])

        tmp_obs_hist.append(pred_obs)
        tmp_act_hist.append(action)
        total_reward+=pred_reward[0,0]

    return total_reward


def f(x):
    #input: action particles (num_particles, step_num*act_dim)
    #output: total rewards

    n_particles = x.shape[0]
    j = [reward_funtion(x[i,:]) for i in range(n_particles)]

    #maximize, so negative
    return -np.asarray(j)




def act(step_num, israndom):
    #random action
    if israndom:
        return np.random.uniform(-0.02,0.02,36)

    #MPC action, sample sample_num trajs, pick the one with highest reward, choose that action
    #solve PSO
    options = {'c1': 0.5, 'c2': 0.6, 'w':0.7}
    dimensions = actor_critic.act_dim*step_num
    bounds = (-0.02*np.ones(dimensions), 0.02*np.ones(dimensions))

    #substitute part of the init value to be the last global optimum
    init_pose = np.random.uniform(-0.02,0.02,(15,dimensions))
    global_opt = []
    if step_num==3:
        global_opt = copy.copy(global_opt_1)
    if step_num==2:
        global_opt = copy.copy(global_opt_2)
    if step_num==1:
        global_opt = copy.copy(global_opt_3)
    for i in range(len(global_opt)):
        init_pose[i,:] = global_opt[i]

    optimizer = ps.single.GlobalBestPSO(n_particles=15, dimensions=dimensions, options=options, bounds = bounds, init_pos = init_pose)

    # Perform optimization
    cost, action = optimizer.optimize(f, iters=7)##modify to iterate more for large space
    first_action = action[0:actor_critic.act_dim]

    if step_num==3:
        global_opt_1.append(action)
    if step_num==2:
        global_opt_2.append(action)
    if step_num==1:
        global_opt_3.append(action)

    return first_action




def normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm+1e-16)

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



def simulation():
    global obs_list
    global act_list
    #record the reward trend
    reward_list = []
    test_reward_list = []
    trans_loss_trend = []
    reward_loss_trend = []

    #read memory
    #memory = np.loadtxt("/home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/memory.txt")
    #actor_critic.memory = deque(memory.tolist())
    #save path:
    actor_checkpoint = rospy.get_param("/rl_client/actor_checkpoint")
    critic_checkpoint = rospy.get_param("/rl_client/critic_checkpoint")
    fig_path = rospy.get_param("/rl_client/figure_path")
    result_path = rospy.get_param("/rl_client/result_path")
    test_result_path = rospy.get_param("/rl_client/test_result_path")

    actor_critic.trans_model.load_weights(actor_checkpoint)
    actor_critic.reward_model.load_weights(critic_checkpoint)
    #init
    obs_dim = 12
    action_dim = 36
    rospy.init_node('RL_client', anonymous=True)
    # sess = tf.compat.v1.Session()
    # K.set_session(sess)

    num_trials = 10000
    trial_len  = 3

    for i in range(num_trials):

        reset()
        cur_state = np.zeros(obs_dim)
        reward_sum = 0
        obs_list = []
        act_list = []
        cur_state = cur_state.reshape((1,obs_dim))
        obs_list.append(cur_state)
        act_list.append(np.zeros((1,action_dim)))

        #initially chose random actions
        israndom = 0
        if i<10:
            israndom = 1

        for j in range(trial_len):
            #env.render()
            print("trial:" + str(i))
            print("step:" + str(j))
            obs_seq = np.asarray(obs_list)
            act_seq = np.asarray(act_list)
            print("act_seq"+str(act_seq))
            obs_seq = obs_seq.reshape((1, -1, obs_dim))
            act_seq = act_seq.reshape((1, -1, action_dim))
            step_num = trial_len-j
            action = act(step_num,israndom)
            action = action.reshape((action_dim))
            cur_state = cur_state.reshape(obs_dim)
            new_state, reward, done = step(cur_state,action)
            reward_sum += reward
            rospy.loginfo(rospy.get_caller_id() + 'got reward %s',reward)
            if j == (trial_len - 1):
                done = True


            #train the agent
            if i%5==0 and len(actor_critic.memory)>32:
                tran_loss,reward_loss = actor_critic.train()
                trans_loss_trend.append(tran_loss)
                reward_loss_trend.append(reward_loss)
                #np.savetxt("/home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/memory.txt",np.array(actor_critic.memory),fmt='%f')

            new_state = np.asarray(new_state).reshape((1,obs_dim))
            action = action.reshape((1,action_dim))

            obs_list.append(new_state)
            act_list.append(action)
            next_obs_seq = np.asarray(obs_list)
            next_act_seq = np.asarray(act_list)
            next_obs_seq = next_obs_seq.reshape((1, -1, obs_dim))
            next_act_seq = next_act_seq.reshape((1, -1, action_dim))

            #padding
            pad_width = trial_len-np.size(obs_seq,1)
            print(pad_width)
            rospy.loginfo(rospy.get_caller_id() + 'obs_shape %s',obs_seq.shape)
            obs_seq = np.pad(obs_seq,((0,0),(pad_width,0),(0,0)),'constant')
            next_obs_seq = np.pad(next_obs_seq,((0,0),(pad_width,0),(0,0)),'constant')
            act_seq = np.pad(act_seq,((0,0),(pad_width,0),(0,0)),'constant')
            next_act_seq = np.pad(next_act_seq,((0,0),(pad_width,0),(0,0)),'constant')
            #print(obs_seq.shape)
            #print(next_obs_seq.shape)

            actor_critic.remember(obs_seq, act_seq, action, reward, next_obs_seq, next_act_seq, done)
            cur_state = new_state
            if done:
                rospy.loginfo(rospy.get_caller_id() + 'got total reward %s',reward_sum)
                reward_list.append(reward_sum)
                break

        if i % 10 == 0 and i>100:
            actor_critic.trans_model.save_weights(actor_checkpoint)
            actor_critic.reward_model.save_weights(critic_checkpoint)
            fig, ax = plt.subplots()
            ax.plot(reward_list)
            fig.savefig(fig_path)
            fig_t, ax_t = plt.subplots()
            ax_t.plot(trans_loss_trend)
            fig_t.savefig('/home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/trans_loss.png')
            fig_r, ax_r= plt.subplots()
            ax_r.plot(reward_loss_trend)
            fig_r.savefig('/home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/reward_loss.png')

            np.savetxt(result_path, reward_list, fmt='%f')

            #test
            reset()
            cur_state = np.zeros(obs_dim)
            test_reward_sum = 0
            test_reward_sum = 0
            obs_list = []
            act_list = []
            cur_state = cur_state.reshape((1,obs_dim))
            obs_list.append(cur_state)
            act_list.append(np.zeros((1,action_dim)))
            for j in range(trial_len):
                #env.render()
                print("test:" + str(i))
                print("step:" + str(j))

                obs_seq = np.asarray(obs_list)
                print("obs_seq"+str(obs_seq))
                act_seq = np.asarray(act_list)
                obs_seq = obs_seq.reshape((1, -1, obs_dim))
                act_seq = act_seq.reshape((1, -1, action_dim))
                step_num = trial_len-j
                action = act(step_num,0)
                action = action.reshape((action_dim))
                cur_state = cur_state.reshape(obs_dim)
                new_state, reward, done = step(cur_state,action)
                test_reward_sum += reward
                rospy.loginfo(rospy.get_caller_id() + 'got reward %s',reward)
                if j == (trial_len - 1):
                    done = True


                new_state = np.asarray(new_state).reshape((1,obs_dim))
                action = action.reshape((1,action_dim))

                obs_list.append(new_state)
                act_list.append(action)
                cur_state = new_state
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
