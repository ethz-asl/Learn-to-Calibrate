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
import pyswarms as ps

# Import sphere function as objective function

# Import backend modules
import pyswarms.backend as P
from pyswarms.backend.topology import Star

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
    def __init__(self,sess):
        self.sess = sess

        self.trans_learning_rate = 1e-4  #1e-4 #5e-5 #5e-5
        self.reward_learning_rate = 1e-4   #1e-4 #5e-5 #5e-5

        self.num_trial = 1
        self.epsilon = .9
        self.epsilon_decay = .9999 #0.999
        self.gamma = .90
        self.tau   = .01
        self.obs_dim = 6
        self.act_dim = 6
        self.num_step = 5
        # ===================================================================== #
        #                               transition Model                             #
        # ===================================================================== #
        tf.compat.v1.disable_eager_execution()
        self.memory = deque(maxlen=10000)



        # ===================================================================== #
        #                             Reward Model                             #
        # ===================================================================== #

        self.reward_act_hist_input, self.reward_action_input, \
        self.reward_model = self.create_reward_model()

        # Initialize for later gradient calculations
        self.sess.run(tf.compat.v1.global_variables_initializer())

        # ===================================================================== #
        #                             return Model                             #
        # ===================================================================== #
        self.return_models = []
        self.init_actions = []
        self.return_act_inputs = []
        self.act_grads = []
        self.total_rs = []
        for i in range(self.num_step):
            init_action, return_act_input, total_r, return_model = self.create_return_model(self.num_step-i)
            act_grad = tf.keras.backend.gradients(total_r,return_act_input)
            self.return_models.append(return_model)
            self.init_actions.append(init_action)
            self.return_act_inputs.append(return_act_input)
            self.total_rs.append(total_r)
            self.act_grads.append(act_grad)





    # ========================================================================= #
    #                              Model Definitions                            #


    def create_reward_model(self):
        act_hist_input = Input(shape=(None,self.act_dim))
        mask_action_input = Masking(mask_value=0.)(act_hist_input)
        #simplified network structure
        reward_rnn,state_h2 = GRU(256, return_state=True)(mask_action_input)

        action_input = Input(shape=self.act_dim)
        action_h2    = Dense(256)(action_input)

        merged    = Concatenate()([state_h2, action_h2])
        merged_h1 = Dense(256, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model  = Model([act_hist_input,action_input],output)

        adam  = Adam(lr=self.reward_learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return act_hist_input, action_input, model

    def create_return_model(self, step_num):
        init_action_input = Input(shape = (None,self.act_dim))
        action_input = Input(shape = (step_num,self.act_dim))
        total_r = 0
        a = init_action_input
        for i in range(step_num):
            r = self.reward_model([a,action_input[:,i,:]])
            action = tf.reshape(action_input[:,i,:],[-1,1,self.act_dim])
            a = Concatenate(axis=1)([a,action])
            total_r+=r

        return_model = Model([init_action_input,action_input],total_r)
        adam  = Adam(lr=self.reward_learning_rate)
        return_model.compile(loss="mse", optimizer=adam)

        return init_action_input,action_input,total_r,return_model

    def compute_gradient(self,step_num,init_action,action_seq):
        k = self.num_step-step_num
        gradients = self.sess.run(self.act_grads[k],feed_dict = {
            self.init_actions[k]:init_action,
            self.return_act_inputs[k]:action_seq
        })
        return gradients
    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_act_hist, action, reward, new_act_hist, done):
        self.memory.append([cur_act_hist, action, reward, new_act_hist, done])


    def _train_reward(self, samples):
        cur_act_hists, actions, rewards,new_act_hists, dones = stack_samples(samples)
        evaluation = self.reward_model.fit([cur_act_hists, actions], rewards, batch_size = 32, epochs = 1, verbose=0)
        print(evaluation.history)
        return evaluation.history['loss']

    def train(self):

        #samples = random.sample(self.memory, len(self.memory)/2)
        samples = np.asarray(self.memory)
        self.samples = samples
        reward_loss = self._train_reward(samples)
        return reward_loss


tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.Session()
K.set_session(sess)
actor_critic = ActorCritic(sess)
act_list = []
global_opt_1 = []
global_opt_2 = []
global_opt_3 = []
global_opt_4 = []


# ========================================================================= #
#                              Model Predictions                            #
# ========================================================================= #


def reward_funtion(x):
    #input: next actions shape (step_num*act_dim,1)
    #output: total reward
    global act_list
    global actor_critic
    step_num = int(x.shape[0]/actor_critic.act_dim)
    tmp_act_hist = copy.copy(act_list)
    total_reward = 0
    for step in range(step_num):
        action = x[step*actor_critic.act_dim:(step+1)*actor_critic.act_dim]

        action = action.reshape((1,-1))
        #print("obs_seq"+str(obs_seq))
        act_seq = np.asarray(tmp_act_hist)
        act_seq = act_seq.reshape((1, -1, actor_critic.act_dim))
        pred_reward = actor_critic.reward_model.predict([act_seq,action])

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

# ========================================================================= #
#                              PSO class                           #
# ========================================================================= #
#include gradients to substitute the local minimum
#Keep select_num particles for the initialization of next problem
def PSO_grad_optimize(n_particles, dimensions,  bounds, init_pose, select_num, iteration, lr, init_act):
    #Input:
    # n_particles:num of particles
    #dimension: dim of varibales
    #bounds: boundary of actions
    #init_pose: initial pose
    #select_num: num of particle kept
    #iteration: number of iterations

    #lr: learning rate of apply gradient
    #init_state: initial state input for gradient compute
    #init_act:  initial action input for gradient compute

    #output:
    #the best pose
    #a list of selected pos

    #define partcicle
    my_topology = Star() # The Topology Class
    my_options = {'c1': 0.1, 'c2': 0.000, 'w': 0.000} # arbitrarily set #0.01,0.01
    my_swarm = P.create_swarm(n_particles=n_particles, dimensions=dimensions, options=my_options, bounds = bounds, init_pos = init_pose) # The Swarm Class

    for i in range(iteration):
        # Part 1: Update personal best
        step_num = int(dimensions/actor_critic.act_dim)
        my_swarm.current_cost = f(my_swarm.position)
        cur_action = my_swarm.position.reshape((n_particles,step_num,actor_critic.act_dim))
        gradients = actor_critic.compute_gradient(step_num,init_act,cur_action)

        my_swarm.pbest_pos = cur_action.reshape((-1,dimensions))+gradients[0].reshape((-1,dimensions))*lr

        my_swarm.pbest_pos = np.clip(my_swarm.pbest_pos,-0.5,0.5)
        my_swarm.pbest_cost = f(my_swarm.pbest_pos)
        print(my_swarm.pbest_cost)

        # Part 2: Update global best
        # Note that gbest computation is dependent on your topology
        if np.min(my_swarm.pbest_cost) < my_swarm.best_cost:
            my_swarm.best_pos, my_swarm.best_cost = my_topology.compute_gbest(my_swarm)

        # Let's print our output

        print('Iteration: {} | my_swarm.best_cost: {:.4f}'.format(i+1, my_swarm.best_cost))

        # Part 3: Update position and velocity matrices
        # Note that position and velocity updates are dependent on your topology
        my_swarm.velocity = my_topology.compute_velocity(my_swarm)
        my_swarm.position = my_topology.compute_position(my_swarm)
        my_swarm.position = np.clip(my_swarm.position,-0.5,0.5)


    best_index = my_swarm.pbest_cost.argsort()[:select_num]
    print(best_index)
    good_actions = my_swarm.pbest_pos[best_index].tolist()
    good_actions.append(my_swarm.best_pos)
    return np.clip(my_swarm.best_pos,-0.5,0.5), np.asarray(good_actions)


def act(step_num, israndom, isdeter):
    global obs_list
    global act_list
    global actor_critic
    global global_opt_1
    global global_opt_2
    global global_opt_3
    global global_opt_4
    #random action
    if israndom or step_num==0:
        return np.random.uniform(-0.5,0.5,5)

    #MPC action, sample sample_num trajs, pick the one with highest reward, choose that action
    #solve PSO
    dimensions = actor_critic.act_dim*step_num
    bounds = (-0.5*np.ones(dimensions), 0.5*np.ones(dimensions))

    #substitute part of the init value to be the last global optimum
    init_pose = np.random.uniform(-0.5,0.5,(15,dimensions))
    global_opt = []
    if step_num==4:
        global_opt = copy.copy(global_opt_1)
    if step_num==3:
        global_opt = copy.copy(global_opt_2)
    if step_num==2:
        global_opt = copy.copy(global_opt_3)
    if step_num==1:
        global_opt = copy.copy(global_opt_4)
    for i in range(len(global_opt)):
        init_pose[i,:] = global_opt[i]

    init_act = np.asarray(act_list).reshape((1,-1,actor_critic.act_dim))
    init_act = np.repeat(init_act,15,axis = 0)
    print(init_act.shape)
    action, good_actions = PSO_grad_optimize(n_particles=15,dimensions=dimensions,
                                             bounds=bounds,init_pose=init_pose,select_num=5,
                                             iteration=5,lr=1e-4,init_act=init_act)
    print(action.shape)
    if isdeter:
        first_action = action[0:actor_critic.act_dim]
    else:
        sel_ind = np.random.choice(np.arange(0,6,1),1)
        rand = np.random.uniform()
        if rand<0.1:
            print("pick"+str(sel_ind))
            first_action = good_actions[sel_ind,0:actor_critic.act_dim]
        else:
            first_action = action[0:actor_critic.act_dim]

    if step_num==4:
        global_opt_1 = good_actions
    if step_num==3:
        global_opt_2 = good_actions
    if step_num==2:
        global_opt_3 = good_actions
    if step_num==1:
        global_opt_4 = good_actions

    return first_action




def normalize(v):
    norm = np.linalg.norm(v)
    return v / (norm+1e-16)

def step(action, last):
    rospy.wait_for_service('/model_client/rl_service')
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    take_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
    resp1 = take_step(0, action.tolist(), last)
    return resp1.reward

def reset():
    rospy.wait_for_service('/model_client/rl_service')
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    reset_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
    resp1 = reset_step(1, [], 0)
    return resp1.reward



def simulation():
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
    train_path = rospy.get_param("/rl_client/train_fig_path")
    result_path = rospy.get_param("/rl_client/result_path")
    test_result_path = rospy.get_param("/rl_client/test_result_path")
    reward_path = rospy.get_param("/rl_client/reward_path")
    trans_path = rospy.get_param("/rl_client/trans_path")
    action_path = rospy.get_param("/rl_client/action_path")
    test_fig_path = rospy.get_param("/rl_client/test_fig_path")

    actor_critic.reward_model.load_weights(critic_checkpoint)
    #init
    action_dim = 6
    rospy.init_node('RL_client', anonymous=True)
    # sess = tf.compat.v1.Session()
    # K.set_session(sess)

    num_trials = 10000
    trial_len  = 5

    for i in range(num_trials):

        reset()
        reward_sum = 0
        act_list = []
        act_list.append(np.zeros((1,action_dim)))
        done = 0

        #initially chose random actions
        israndom = 0
        if i<0:
            israndom = 1

        for j in range(trial_len):
            #env.render()
            print("trial:" + str(i))
            print("step:" + str(j))
            act_seq = np.asarray(act_list)
            print("act_seq"+str(act_seq))
            act_seq = act_seq.reshape((1, -1, action_dim))
            step_num = trial_len-j-1
            action = act(step_num,israndom,0)
            action = action.reshape((action_dim))
            last = 0
            if j == (trial_len - 1):
                last = 1
            reward = step(action,last)
            reward_sum += reward
            rospy.loginfo(rospy.get_caller_id() + 'got reward %s',reward)
            if j == (trial_len - 1):
                done = True


            #train the agent
            if len(actor_critic.memory)>1:
                reward_loss = actor_critic.train()
                reward_loss_trend.append(reward_loss)
                #np.savetxt("/home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/memory.txt",np.array(actor_critic.memory),fmt='%f')

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

        if i % 10 == 0 and i>0:
            actor_critic.reward_model.save_weights(critic_checkpoint)
            fig, ax = plt.subplots()
            ax.plot(reward_list)
            fig.savefig(train_path)
            fig_t, ax_t = plt.subplots()
            ax_t.plot(trans_loss_trend)
            fig_t.savefig(trans_path)
            fig_r, ax_r= plt.subplots()
            ax_r.plot(reward_loss_trend)
            fig_r.savefig(reward_path)

            np.savetxt(result_path, reward_list, fmt='%f')

            #test
            reset()
            test_reward_sum = 0
            test_reward_sum = 0
            act_list = []
            act_list.append(np.zeros((1,action_dim)))
            done = 0
            for j in range(trial_len):
                #env.render()
                print("test:" + str(i))
                print("step:" + str(j))

                act_seq = np.asarray(act_list)
                act_seq = act_seq.reshape((1, -1, action_dim))
                step_num = trial_len-j-1
                action = act(step_num,0,1)
                action = action.reshape((action_dim))
                last = 0
                if j == (trial_len - 1):
                    last = 1
                reward = step(action,last)
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
                    fig1.savefig(test_fig_path)
                    np.savetxt(test_result_path, test_reward_list, fmt='%f')
                    np.savetxt(action_path, np.asarray(act_list).reshape((-1,action_dim)), fmt='%f')
                    break



if __name__ == '__main__':
    simulation()
