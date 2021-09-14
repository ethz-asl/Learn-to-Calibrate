#!/usr/bin/env python


import rospy
from franka_cal_sim.srv import*
import cv2
from cv_bridge import CvBridge
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, GRU, Masking, LSTM
from tensorflow.keras.layers import Add, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.compat.v1.keras import backend as K
import datetime
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tempfile import TemporaryFile

import random
from collections import deque

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
    def __init__(self, sess):
        self.sess = sess

        self.critic_learning_rate = 1e-4  #1e-4 #5e-5 #5e-5
        self.actor_learning_rate = 1e-5   #1e-4 #5e-5 #5e-5
        self.batch_size = 64
        self.explore_rate = 0.002 #0.005  #0.003 #0.002
        self.num_trial = 1
        self.epsilon = .9
        self.epsilon_decay = .9999 #0.999
        self.gamma = .99
        self.tau   = 0.01
        self.obs_dim = 12
        self.act_dim = 36
        self.reg_w = 0.01

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #
        tf.compat.v1.disable_eager_execution()
        self.ema = tf.train.ExponentialMovingAverage(decay=1-self.tau)
        self.outfile = TemporaryFile()
        self.memory = deque(maxlen=4000)
        self.actor_state_input, self.actor_act_hist_input, self.actor_model = self.create_actor_model()
        _,_, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.compat.v1.placeholder(tf.float32,
                                                [None, self.act_dim]) # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        ################remove minus here
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.compat.v1.train.AdamOptimizer(self.actor_learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_act_hist_input, self.critic_action_input, \
        self.critic_model = self.create_critic_model()
        _, _, _,self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input) # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.compat.v1.global_variables_initializer())

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

    def create_actor_model(self):
        # ========================================================================= #
        state_input = Input(shape=(None,self.obs_dim))
        act_hist_input = Input(shape=(None,self.act_dim))
        mask_state_input = Masking(mask_value=0.)(state_input)
        mask_action_input = Masking(mask_value=0.)(act_hist_input)
        # obs_h1 = Dense(64, activation='relu')(mask_state_input)
        # act_h1 = Dense(64, activation='relu')(mask_action_input)
        actor_rnn,state_h,state_h_1 =LSTM(64, return_state=True)(Concatenate()([mask_state_input,mask_action_input]))
        #state_h = Dense(64, return_state=True)(Concatenate()([obs_h1,act_h1]))
        h2 = Dense(256, activation='relu')(actor_rnn)
        output = Dense(self.act_dim, activation='tanh')(h2)

        model = Model([state_input,act_hist_input], output)
        adam  = Adam(lr=0.0001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, act_hist_input, model

    def create_critic_model(self):
        state_input = Input(shape=(None,self.obs_dim))
        act_hist_input = Input(shape=(None,self.act_dim))
        mask_state_input = Masking(mask_value=0.)(state_input)
        mask_action_input = Masking(mask_value=0.)(act_hist_input)
        # state_h1 = Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(self.reg_w))(mask_state_input)
        # action_h1 = Dense(64, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(self.reg_w))(mask_action_input)
        critic_rnn,state_h2,state_c2 = LSTM(64, return_state=True,kernel_regularizer=tf.keras.regularizers.l2(self.reg_w))(Concatenate()([mask_state_input,mask_action_input]))

        action_input = Input(shape=self.act_dim)
        action_h1    = Dense(64,kernel_regularizer=tf.keras.regularizers.l2(self.reg_w))(action_input)

        merged    = Concatenate()([critic_rnn, action_h1])
        merged_h1 = Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(self.reg_w))(merged)
        output = Dense(1, activation='linear',kernel_regularizer=tf.keras.regularizers.l2(self.reg_w))(merged_h1)
        model  = Model([state_input,act_hist_input,action_input],output)

        adam  = Adam(lr=self.critic_learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, act_hist_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, cur_act_hist, action, reward, new_state, new_act_hist, done):
        self.memory.append([cur_state, cur_act_hist, action, reward, new_state, new_act_hist, done])

    def _train_actor(self, samples):

        cur_states, cur_act_hists, actions, rewards, new_states,new_act_hists, _ =  stack_samples(samples)
        predicted_actions = self.actor_model.predict([cur_states,cur_act_hists])
        grads = self.sess.run(self.critic_grads, feed_dict={
            self.critic_state_input:  cur_states,
            self.critic_act_hist_input: cur_act_hists,
            self.critic_action_input: predicted_actions
        })[0]
        print(grads)
        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: cur_states,
            self.actor_act_hist_input: cur_act_hists,
            self.actor_critic_grad: grads
        })

        # #save
        # status = self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        #
        # status.assert_consumed()  # Optional sanity checks.
        # self.checkpoint.save(file_prefix=self.checkpoint_prefix)


    def _train_critic(self, samples):
        cur_states, cur_act_hists, actions, rewards, new_states,new_act_hists, dones = stack_samples(samples)
        target_actions = self.target_actor_model.predict([new_states,new_act_hists])
        future_rewards = self.target_critic_model.predict([new_states, new_act_hists,target_actions])
        dones = dones.reshape(rewards.shape)
        future_rewards = future_rewards.reshape(rewards.shape)
        rewards += self.gamma * future_rewards * (1 - dones)

        #modified to restrict target value
        num_ls = np.sum(abs(target_actions)>0.9,axis=1)
        #print(1-dones)
        # print(target_actions)
        # print(future_rewards)
        print(num_ls)
        #rewards=rewards-num_ls.reshape((-1,1))*0.5

        evaluation = self.critic_model.fit([cur_states, cur_act_hists, actions], rewards, verbose=0)
        print(evaluation.history)

    def train(self,i):
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        self.samples = samples
        self._train_critic(samples)
        self._train_actor(samples)
        # if i%200<100:
        #     self._train_critic(samples)
        # if i%200>100:
        #     self._train_actor(samples)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]*self.tau + actor_target_weights[i]*(1-self.tau)
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
        self.target_critic_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state, cur_act_hist):
        self.epsilon *= self.epsilon_decay
        rospy.loginfo(rospy.get_caller_id() + 'decay %s',self.epsilon)
        if np.random.random() < self.epsilon:
            ##Change to normal, random or normal is better?
            #0.005
            return np.random.normal(size=(36,))*self.explore_rate+self.actor_model.predict([cur_state,cur_act_hist])
        return self.actor_model.predict([cur_state,cur_act_hist])

def read_replay(path,actor,obs_dim,action_dim):
    f = open(path, "r")
    obs, actions, rewards, next_obs, terminals = [],[],[],[],[]
    for line in f:
        cols = line.strip().split('\t')
        obs_s = cols[0].split(';')
        obs_t = []
        for obss in obs_s:
            obs_t.append([float(v) for v in obss.split(',')])
        obs_s_tp1 = cols[3].split(';')
        obs_tp1 = []
        for obss in obs_s_tp1:
            obs_tp1.append([float(v) for v in obss.split(',')])
        obs_t = np.array(obs_t)
        obs.append(obs_t)
        action = np.array([float(v) for v in cols[1].split(',')])
        actions.append(action)
        rewards.append(float(cols[2]))
        obs_tp1 = np.array(obs_tp1)
        next_obs.append(obs_tp1)
        terminals.append(bool(cols[4]=="True"))

    print(terminals)
    i = np.argmax(rewards)
    print(i)
    print(obs[i])
    print(actions[i])
    print(rewards[i])
    plt.plot(np.linalg.norm(actions,axis=1))
    plt.show()
    plt.plot(rewards)
    plt.show()

    #save data
    for i in range(len(obs)):
        obs_action_seq = obs[i]
        obs_seq = obs_action_seq[:,0:12]
        ########
        #obs_seq = np.zeros((6,12))
        act_seq = obs_action_seq[:,12:48]
        #print(obs_seq)
        #print(act_seq)
        num_l = np.sum(abs(actions[i])>0.018)
        action = actions[i]
        if np.linalg.norm(action)<1:
            action = actions[i]/0.02



        reward = rewards[i]
        #reward-=num_l*0.5

        next_obs_action_seq = next_obs[i]
        next_obs_seq = next_obs_action_seq[:,0:12]
        next_act_seq = next_obs_action_seq[:,12:48]
        obs_seq = obs_seq.reshape((1, -1, obs_dim))
        act_seq = act_seq.reshape((1, -1, action_dim))
        #next_obs_seq = np.zeros((6,12))
        next_obs_seq = next_obs_seq.reshape((1, -1, obs_dim))
        ########
        next_act_seq = next_act_seq.reshape((1, -1, action_dim))
        done = terminals[i]
        actor.remember(obs_seq, act_seq, action, reward, next_obs_seq, next_act_seq, done)
        #augment data
        next_act_seq_1 = np.concatenate([act_seq.reshape(((-1,action_dim)))[1:],-action.reshape((1,action_dim))],axis = 0).reshape((1, -1, action_dim))
        next_act_seq_2 = np.concatenate([-act_seq.reshape(((-1,action_dim)))[1:],action.reshape((1,action_dim))],axis = 0).reshape((1, -1, action_dim))
        actor.remember(obs_seq, -act_seq, -action, reward, next_obs_seq, -next_act_seq, done)
        actor.remember(obs_seq, act_seq, -action, reward, next_obs_seq, next_act_seq_1, done)
        actor.remember(obs_seq, -act_seq, action, reward, next_obs_seq, next_act_seq_2, done)

    print(len(obs))


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

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
    #record the reward trend
    reward_list = []
    test_reward_list = []

    #save path:
    # actor_checkpoint = rospy.get_param("/rl_client/actor_checkpoint")
    # critic_checkpoint = rospy.get_param("/rl_client/critic_checkpoint")
    # fig_path = rospy.get_param("/rl_client/figure_path")
    # result_path = rospy.get_param("/rl_client/result_path")
    # test_result_path = rospy.get_param("/rl_client/test_result_path")

    #init
    obs_dim = 12
    action_dim = 36
    sess = tf.compat.v1.Session()
    K.set_session(sess)
    actor_critic = ActorCritic(sess)

    path = "/home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/replay_buffers/replay_buffer_imu_final.txt"
    #read replay buffer
    read_replay(path,actor_critic,obs_dim,action_dim)

    path = "/home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/replay_buffer_last.txt"
    #read replay buffer
    read_replay(path,actor_critic,obs_dim,action_dim)



    path = "/home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/replay_buffers/replay_buffer_tiny.txt"

    read_replay(path,actor_critic,obs_dim,action_dim)


    num_trials = 10000
    trial_len  = 3

    for i in range(num_trials):

        # reset()
        # cur_state = np.ones(obs_dim)*500
        # reward_sum = 0
        # obs_list = []
        # act_list = []
        # cur_state = cur_state.reshape((1,obs_dim))
        # obs_list.append(normalize(cur_state))
        # act_list.append(normalize(0.01*np.ones((1,action_dim))))
        # for j in range(trial_len):
        #     #env.render()
        #     print("trial:" + str(i))
        #     print("step:" + str(j))
        #     obs_seq = np.asarray(obs_list)
        #     print("obs_seq"+str(obs_seq))
        #     act_seq = np.asarray(act_list)
        #     obs_seq = obs_seq.reshape((1, -1, obs_dim))
        #     act_seq = act_seq.reshape((1, -1, action_dim))
        #     action = actor_critic.act(obs_seq,act_seq)
        #     action = action.reshape((action_dim))
        #     cur_state = cur_state.reshape(obs_dim)
        #     new_state, reward, done = step(cur_state,action)
        #     reward_sum += reward
        #     rospy.loginfo(rospy.get_caller_id() + 'got reward %s',reward)
        #     if j == (trial_len - 1):
        #         done = True



            actor_critic.train(i)
            actor_critic.update_target()
            obs_list = []
            act_list = []
            cur_state = np.zeros(obs_dim)
            cur_state = cur_state.reshape((1,obs_dim))
            obs_list.append(cur_state)
            act_list.append(0.01*np.ones((1,action_dim)))
            obs_seq = np.asarray(obs_list)
            act_seq = np.asarray(act_list)
            obs_seq = obs_seq.reshape((1, -1, obs_dim))
            act_seq = act_seq.reshape((1, -1, action_dim))
            action = actor_critic.actor_model.predict([obs_seq,act_seq])*0.02
            if i%200==0:
                print(action[0])
                print(np.linalg.norm(action[0]))
                action = np.ones((1,action_dim))
                print(actor_critic.critic_model.predict([obs_seq,act_seq,action]))
                action = np.ones((1,action_dim))*0.05
                act_seq = np.ones((1,1,action_dim))*0.001
                print(actor_critic.critic_model.predict([obs_seq,act_seq,action]))

            # new_state = np.asarray(new_state).reshape((1,obs_dim))
            # action = action.reshape((1,action_dim))

            # obs_list.append(normalize(new_state))
            # act_list.append(normalize(action))
            # next_obs_seq = np.asarray(obs_list)
            # next_act_seq = np.asarray(act_list)
            # next_obs_seq = next_obs_seq.reshape((1, -1, obs_dim))
            # next_act_seq = next_act_seq.reshape((1, -1, action_dim))

            # #padding
            # pad_width = trial_len-np.size(obs_seq,1)
            # rospy.loginfo(rospy.get_caller_id() + 'obs_shape %s',obs_seq.shape)
            # obs_seq = np.pad(obs_seq,((0,0),(pad_width,0),(0,0)),'constant')
            # next_obs_seq = np.pad(next_obs_seq,((0,0),(pad_width,0),(0,0)),'constant')
            # act_seq = np.pad(act_seq,((0,0),(pad_width,0),(0,0)),'constant')
            # next_act_seq = np.pad(next_act_seq,((0,0),(pad_width,0),(0,0)),'constant')
            # #print(obs_seq.shape)
            # #print(next_obs_seq.shape)

            # actor_critic.remember(obs_seq, act_seq, action, reward, next_obs_seq, next_act_seq, done)
            # cur_state = new_state
            # if done:
            #     rospy.loginfo(rospy.get_caller_id() + 'got total reward %s',reward_sum)
            #     reward_list.append(reward_sum)
            #     break

        # if i % 5 == 0:
        #     actor_critic.actor_model.save_weights(actor_checkpoint)
        #     actor_critic.critic_model.save_weights(critic_checkpoint)
        #     fig, ax = plt.subplots()
        #     ax.plot(reward_list)
        #     fig.savefig(fig_path)
        #     np.savetxt(result_path, reward_list, fmt='%f')
        #
    #draw a graph
    obs_list = []
    act_list = []
    cur_state = np.zeros(obs_dim)
    cur_state = cur_state.reshape((1,obs_dim))
    obs_list.append(cur_state)
    act_list.append(0.01*np.ones((1,action_dim)))
    obs_seq = np.asarray(obs_list)
    act_seq = np.asarray(act_list)
    obs_seq = obs_seq.reshape((1, -1, obs_dim))
    act_seq = act_seq.reshape((1, -1, action_dim))
    q = []
    t_q = []
    for i in range(100):
        action = 0.015*np.ones((1,action_dim))*i
        q_value = actor_critic.critic_model.predict([obs_seq,act_seq,action])
        t_q_value = actor_critic.target_critic_model.predict([obs_seq,act_seq,action])
        t_q.append(t_q_value.reshape((-1,)))
        q.append(q_value.reshape((-1,)))

    plt.plot(q)
    plt.show()
    plt.plot(t_q)
    plt.show()
    # #test
    # rospy.init_node('RL_client', anonymous=True)
    # reset()
    # cur_state = np.zeros(obs_dim)
    # test_reward_sum = 0
    # test_reward_sum = 0
    # obs_list = []
    # act_list = []
    # cur_state = cur_state.reshape((1,obs_dim))
    # obs_list.append(cur_state)
    # act_list.append(0.01*np.ones((1,action_dim)))
    # for j in range(trial_len):
    #     #env.render()
    #     print("test:" + str(i))
    #     print("step:" + str(j))
    #
    #     obs_seq = np.asarray(obs_list)
    #     print("obs_seq:" + str(obs_seq))
    #     act_seq = np.asarray(act_list)
    #     #obs_seq = np.zeros((6,12))
    #     obs_seq = obs_seq.reshape((1, -1, obs_dim))
    #
    #     act_seq = act_seq.reshape((1, -1, action_dim))
    #     action = actor_critic.actor_model.predict([obs_seq,act_seq])*0.02
    #     action = action.reshape((action_dim))
    #     print(action)
    #     cur_state = cur_state.reshape(obs_dim)
    #     new_state, reward, done = step(cur_state,action)
    #     test_reward_sum += reward
    #     rospy.loginfo(rospy.get_caller_id() + 'got reward %s',reward)
    #     if j == (trial_len - 1):
    #         done = True
    #
    #
    #     new_state = np.asarray(new_state).reshape((1,obs_dim))
    #     action = action.reshape((1,action_dim))
    #
    #     obs_list.append(new_state)
    #     act_list.append(action)
    #     cur_state = new_state
    #     if done:
    #         rospy.loginfo(rospy.get_caller_id() + 'got total reward %s',test_reward_sum)
    #         test_reward_list.append(test_reward_sum)
    #         # fig1, ax1 = plt.subplots()
    #         # ax1.plot(test_reward_list)
    #         #fig1.savefig('/home/yunke/prl_proj/panda_ws/src/franka_cal_sim/python/test_reward.png')
    #         #np.savetxt(test_result_path, test_reward_list, fmt='%f')
    #         break



if __name__ == '__main__':
    simulation()
