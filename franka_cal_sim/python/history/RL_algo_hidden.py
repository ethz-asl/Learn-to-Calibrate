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

def stack_samples(samples):
    array = np.array(samples)

    current_states = np.stack(array[:,0]).reshape((array.shape[0],-1,array[0,0].shape[2]))
    current_act_hists = np.stack(array[:,1]).reshape((array.shape[0],-1,array[0,1].shape[2]))
    actions = np.stack(array[:,2]).reshape((array.shape[0],-1))
    rewards = np.stack(array[:,3]).reshape((array.shape[0],-1))
    new_states = np.stack(array[:,4]).reshape((array.shape[0],-1,array[0,4].shape[2]))
    new_act_hists = np.stack(array[:,5]).reshape((array.shape[0],-1,array[0,5].shape[2]))
    hiddens = np.stack(array[:,6]).reshape((array.shape[0],array[0,6].shape[1]))
    new_hiddens = np.stack(array[:,7]).reshape((array.shape[0],array[0,7].shape[1]))
    dones = np.stack(array[:,8]).reshape((array.shape[0],1))

    return current_states, current_act_hists, actions, rewards, new_states, new_act_hists, hiddens, new_hiddens, dones

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
    def __init__(self, sess):
        self.sess = sess

        self.critic_learning_rate = 1e-4  #1e-4 #5e-5 #5e-5
        self.actor_learning_rate = 1e-4   #1e-4 #5e-5 #5e-5
        self.batch_size = 32
        self.explore_rate = 0.002 #0.005  #0.003 #0.002
        self.num_trial = 1
        self.epsilon = .9
        self.epsilon_decay = .999 #0.999
        self.gamma = .90
        self.tau   = .01
        self.obs_dim = 12
        self.act_dim = 36
        self.hidden_dim = 64

        #data flow
        #last_hidden, obs, last_action -> hidden
        #hidden, action -> Q -> value
        #hidden -> policy -> action
        #next_obs, action, hidden -> next hidden
        tf.compat.v1.disable_eager_execution()
        self.ema = tf.train.ExponentialMovingAverage(decay=1-self.tau)
        self.outfile = TemporaryFile()
        self.memory = deque(maxlen=5000)
        # ===================================================================== #
        #                              hidden Model                             #
        # ===================================================================== #
        self.obs_input,self.last_act_input,self.hidden_input,self.hidden_state,self.hidden_model = self.create_hidden_model()
        self.target_obs_input,self.target_last_act_input,self.target_hidden_input,self.target_hidden_state,self.target_hidden_model = self.create_hidden_model()

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

        self.actor_model = self.create_actor_model()
        self.target_actor_model = self.create_target_actor_model()

        self.actor_critic_grad = tf.compat.v1.placeholder(tf.float32,
                                                [None, self.act_dim]) # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)

        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.compat.v1.train.AdamOptimizer(self.actor_learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_action_input, self.critic_model = self.create_critic_model()
        self.target_critic_act_input,self.target_critic_model = self.create_target_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input) # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.compat.v1.global_variables_initializer())



    # ========================================================================= #
    #                              Model Definitions                            #
    def create_hidden_model(self):
        # ========================================================================= #
        #state, last hidden -> hidden
        state_input = Input(shape=(None,self.obs_dim))
        act_input = Input(shape=(None,self.act_dim))
        hidden_input = Input(shape=(self.hidden_dim,))
        hidden_rnn,state_h = GRU(self.hidden_dim, return_state=True)(inputs=Concatenate()([state_input,act_input]),initial_state=hidden_input)

        model = Model([state_input,act_input,hidden_input], state_h)
        return state_input, act_input, hidden_input, state_h, model



    def create_actor_model(self):
        # ========================================================================= #
        #hidden -> action
        h = Dense(500, activation='relu')(self.hidden_state)
        output = Dense(self.act_dim, activation='linear')(h)
        output_clip = tf.clip_by_value(output, -1, 1)
        model = Model([self.obs_input,self.last_act_input,self.hidden_input], output_clip)
        adam  = Adam(lr=0.0001)
        model.compile(loss="mse", optimizer=adam)
        return model

    def create_target_actor_model(self):
        # ========================================================================= #
        #hidden -> action
        h = Dense(500, activation='relu')(self.target_hidden_state)
        output = Dense(self.act_dim, activation='linear')(h)
        output_clip = tf.clip_by_value(output, -1, 1)
        model = Model([self.target_obs_input,self.target_last_act_input,self.target_hidden_input], output_clip)
        adam  = Adam(lr=0.0001)
        model.compile(loss="mse", optimizer=adam)
        return model

    def create_critic_model(self):
        #hidden,action -> value
        action_input = Input(shape=self.act_dim)
        action_h1    = Dense(500)(action_input)

        merged    = Concatenate()([self.hidden_state, action_h1])
        merged_h1 = Dense(500, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model  = Model([self.obs_input,self.last_act_input,self.hidden_input,action_input],output)

        adam  = Adam(lr=self.critic_learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return action_input, model

    def create_target_critic_model(self):
        #hidden,action -> value
        target_action_input = Input(shape=self.act_dim)
        action_h1    = Dense(500)(target_action_input)

        merged    = Concatenate()([self.target_hidden_state, action_h1])
        merged_h1 = Dense(500, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model  = Model([self.target_obs_input,self.target_last_act_input,self.target_hidden_input,target_action_input],output)

        adam  = Adam(lr=self.critic_learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return target_action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, cur_act_hist, action, reward, new_state, new_act_hist, hidden, new_hidden, done):
        self.memory.append([cur_state, cur_act_hist, action, reward, new_state, new_act_hist, hidden, new_hidden, done])

    def _train_actor(self, samples):

        cur_states, cur_act_hists, actions, rewards, new_states,new_act_hists, hiddens, new_hiddens, _ =  stack_samples(samples)
        predicted_actions = self.actor_model.predict([cur_states,cur_act_hists, hiddens])
        grads = self.sess.run(self.critic_grads, feed_dict={
            self.obs_input:  cur_states,
            self.last_act_input: cur_act_hists,
            self.hidden_input: hiddens,
            self.critic_action_input: predicted_actions
        })[0]

        self.sess.run(self.optimize, feed_dict={
            self.obs_input: cur_states,
            self.last_act_input: cur_act_hists,
            self.hidden_input: hiddens,
            self.actor_critic_grad: grads
        })

        # #save
        # status = self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        #
        # status.assert_consumed()  # Optional sanity checks.
        # self.checkpoint.save(file_prefix=self.checkpoint_prefix)


    def _train_critic(self, samples):
        cur_states, cur_act_hists, actions, rewards, new_states,new_act_hists, hiddens, new_hiddens, dones = stack_samples(samples)
        target_actions = self.target_actor_model.predict([new_states,new_act_hists,new_hiddens])
        future_rewards = self.target_critic_model.predict([new_states, new_act_hists,new_hiddens,target_actions])
        dones = dones.reshape(rewards.shape)
        future_rewards = future_rewards.reshape(rewards.shape)
        rewards += self.gamma * future_rewards * (1 - dones)

        evaluation = self.critic_model.fit([cur_states, cur_act_hists, hiddens, actions], rewards, verbose=0)
        #print(evaluation.history)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        self.samples = samples
        self._train_critic(samples)
        self._train_actor(samples)

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

    def act(self, cur_state, cur_act_hist, hidden):
        self.epsilon *= self.epsilon_decay
        rospy.loginfo(rospy.get_caller_id() + 'decay %s',self.epsilon)
        if np.random.random() < self.epsilon:
            ##Change to normal, random or normal is better?
            #0.005
            return np.random.normal(size=(36,))*self.explore_rate+self.actor_model.predict([cur_state,cur_act_hist,hidden])*0.02
        return self.actor_model.predict([cur_state,cur_act_hist,hidden])*0.02


    def deter_act(self, cur_state, cur_act_hist, hidden):
        return self.actor_model.predict([cur_state,cur_act_hist,hidden])*0.02


    def get_new_hidden(self, cur_state, cur_act_hist, hidden):
        new_hidden = self.sess.run(self.hidden_state,feed_dict = {
            self.obs_input: cur_state,
            self.last_act_input: cur_act_hist,
            self.hidden_input: hidden
        })
        return new_hidden

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
    actor_checkpoint = rospy.get_param("/rl_client/actor_checkpoint")
    critic_checkpoint = rospy.get_param("/rl_client/critic_checkpoint")
    fig_path = rospy.get_param("/rl_client/figure_path")
    result_path = rospy.get_param("/rl_client/result_path")
    test_result_path = rospy.get_param("/rl_client/test_result_path")

    #init
    obs_dim = 12
    action_dim = 36
    rospy.init_node('RL_client', anonymous=True)
    sess = tf.compat.v1.Session()
    K.set_session(sess)
    actor_critic = ActorCritic(sess)

    num_trials = 5000
    trial_len  = 3

    for i in range(num_trials):

        reset()
        cur_state = np.zeros((1,1,obs_dim))
        last_act = np.zeros((1,1,action_dim))
        reward_sum = 0
        hidden = np.zeros((1,actor_critic.hidden_dim))
        print(hidden.shape)
        for j in range(trial_len):
            #env.render()
            print("trial:" + str(i))
            print("step:" + str(j))
            print("state:" + str(cur_state))
            print("hidden:" + str(hidden))
            print("last_action:" + str(last_act))

            #interact
            action = actor_critic.act(cur_state,last_act,hidden)
            action = action.reshape((action_dim))
            cur_state = cur_state.reshape(obs_dim)
            new_state, reward, done = step(cur_state,action)
            reward_sum += reward
            rospy.loginfo(rospy.get_caller_id() + 'got reward %s',reward)
            if j == (trial_len - 1):
                done = True


            #train
            actor_critic.train()
            actor_critic.update_target()

            #remember
            cur_state = cur_state.reshape((1,1,obs_dim))
            new_state = np.asarray(new_state).reshape((1,1,obs_dim))
            last_act = last_act.reshape((1,1,action_dim))
            new_hidden = actor_critic.get_new_hidden(cur_state,last_act,hidden)

            actor_critic.remember(cur_state, last_act, action, reward, new_state, action.reshape((1,1,action_dim)),hidden, new_hidden,done)

            #updata
            cur_state = new_state
            last_act = action.reshape((1,1,action_dim))
            hidden = new_hidden

            if done:
                rospy.loginfo(rospy.get_caller_id() + 'got total reward %s',reward_sum)
                reward_list.append(reward_sum)
                break

        if i % 5 == 0:
            actor_critic.actor_model.save_weights(actor_checkpoint)
            actor_critic.critic_model.save_weights(critic_checkpoint)
            fig, ax = plt.subplots()
            ax.plot(reward_list)
            fig.savefig(fig_path)
            np.savetxt(result_path, reward_list, fmt='%f')

            #test
            reset()
            cur_state = np.zeros((1,1,obs_dim))
            last_act = np.zeros((1,1,action_dim))
            test_reward_sum = 0
            hidden = np.zeros((1,actor_critic.hidden_dim))
            for j in range(trial_len):
                #env.render()
                print("trial:" + str(i))
                print("step:" + str(j))

                #interact
                action = actor_critic.deter_act(cur_state,last_act,hidden)
                action = action.reshape((action_dim))
                cur_state = cur_state.reshape(obs_dim)
                new_state, reward, done = step(cur_state,action)
                test_reward_sum += reward
                rospy.loginfo(rospy.get_caller_id() + 'got reward %s',reward)
                if j == (trial_len - 1):
                    done = True

                #predict
                cur_state = cur_state.reshape((1,1,obs_dim))
                new_state = np.asarray(new_state).reshape((1,1,obs_dim))
                last_act = last_act.reshape((1,1,action_dim))
                new_hidden = actor_critic.get_new_hidden(cur_state,last_act,hidden)

                #update
                cur_state = new_state
                last_act = action.reshape((1,1,action_dim))
                hidden = new_hidden

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
