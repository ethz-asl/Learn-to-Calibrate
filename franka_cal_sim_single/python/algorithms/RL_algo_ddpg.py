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
import gym

import random
from collections import deque

#visualization
import wandb
wandb.init(project="my-project")
wandb.config["more"] = "custom"

def stack_samples(samples):
    data = np.array(samples)

    states = np.stack(data[:,0]).reshape((data.shape[0],-1))
    actions = np.stack(data[:,1]).reshape((data.shape[0],-1))
    rewards = np.stack(data[:,2]).reshape((data.shape[0],-1))
    new_states = np.stack(data[:,3]).reshape((data.shape[0],-1))
    dones = np.stack(data[:,4]).reshape((data.shape[0],1))

    return states, actions, rewards, new_states, dones

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
    def __init__(self, sess):
        self.sess = sess

        self.critic_learning_rate = 1e-4  #1e-4 #5e-5 #5e-5
        self.actor_learning_rate = 1e-4   #1e-4 #5e-5 #5e-5
        self.batch_size = 128
        self.explore_rate = 0.1 #0.005  #0.003 #0.002
        self.num_trial = 50000
        self.epsilon = .99
        self.epsilon_decay = .995 #0.999
        self.gamma = .99
        self.tau   = .01
        self.obs_dim = 7
        self.act_dim = 6

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
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

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

        self.critic_state_input, self.critic_action_input, \
        self.critic_model = self.create_critic_model()
        _, _,self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(tf.reduce_mean(self.critic_model.output),
                                         self.critic_action_input) # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.compat.v1.global_variables_initializer())



    # ========================================================================= #
    #                              Model Definitions                            #

    def create_actor_model(self):
        # ========================================================================= #
        state_input = Input(shape=(self.obs_dim))
        h1 = Dense(256, activation='relu')(state_input)
        h2 = Dense(128, activation='relu')(h1)
        output = Dense(self.act_dim, activation='tanh')(h2)

        model = Model([state_input], output)
        adam  = Adam(lr=self.actor_learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=(self.obs_dim))
        
        action_input = Input(shape=self.act_dim)

        merged    = Concatenate()([state_input, action_input])
        merged_h1 = Dense(256, activation='relu')(merged)
        merged_h2 = Dense(128, activation='relu')(merged_h1)
        output = Dense(1, activation='linear')(merged_h2)
        model  = Model([state_input,action_input],output)

        adam  = Adam(lr=self.critic_learning_rate)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_act_hist, action, reward, new_act_hist, done):
        self.memory.append([cur_act_hist, action, reward, new_act_hist, done])

    def _train_actor(self, samples):

        states, actions, rewards, new_states, _ =  stack_samples(samples)
        predicted_actions = self.actor_model.predict([states])
        grads = self.sess.run(self.critic_grads, feed_dict={
            self.critic_state_input: states,
            self.critic_action_input: predicted_actions
        })[0]

        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: states,
            self.actor_critic_grad: grads
        })
        
        #return actor loss
        predicted_Q = self.critic_model.predict([states, predicted_actions])
        return -np.mean(predicted_Q)

        # #save
        # status = self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        #
        # status.assert_consumed()  # Optional sanity checks.
        # self.checkpoint.save(file_prefix=self.checkpoint_prefix)


    def _train_critic(self, samples):
        states, actions, rewards, new_states, dones = stack_samples(samples)
        target_actions = self.target_actor_model.predict([new_states])
        future_rewards = self.target_critic_model.predict([new_states,target_actions])
        dones = dones.reshape(rewards.shape)
        future_rewards = future_rewards.reshape(rewards.shape)
        rewards += self.gamma * np.multiply(future_rewards,(1 - dones))

        evaluation = self.critic_model.fit([states, actions], rewards, verbose=0)
        #print(evaluation.history)

        #return critic loss
        predicted_Q = self.critic_model.predict([states,actions])
        return np.mean((predicted_Q-rewards)**2)
        

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        self.samples = samples
        critic_loss = self._train_critic(samples)
        actor_loss = self._train_actor(samples)

        return critic_loss, actor_loss

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

    def act(self, cur_state, is_test):
        self.epsilon *= self.epsilon_decay
        #network out put:
        raw_action = self.actor_model.predict(cur_state)
        if not is_test:
            noise = np.random.normal(size=(self.act_dim,))
            raw_action = np.clip(raw_action+noise*self.explore_rate,-np.ones((6,)),np.ones((6,)))
             
        rospy.loginfo(rospy.get_caller_id() + 'decay %s',self.epsilon)
        if np.random.random() < self.epsilon:
            ##Change to normal, random or normal is better?
            raw_action = np.random.uniform(-np.ones((6,)),np.ones((6,)))

        
        #scale of actions
        scale = np.asarray([0.3,1.5,1.5,0.4,0.4,0.4])    
        det_action = np.multiply(raw_action,scale)

        return raw_action,det_action


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def step(action, last):
    rospy.wait_for_service('/model_client/rl_service')
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    take_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
    
    try:
        resp1 = take_step(0, action.tolist(), last)
    except:
        return np.zeros((6,)), 0
    reward = resp1.reward - np.linalg.norm(action)*0.02
    return resp1.next_state, reward

def reset():
    rospy.wait_for_service('/model_client/rl_service')
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    reset_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
    try:
        resp1 = reset_step(1, [], 0)
    except:
        return 0
    return resp1.reward



def simulation():
    #record the reward trend
    reward_list = []
    test_reward_list = []
    critic_loss_list = []
    actor_loss_list = []

    #save path:
    actor_checkpoint = rospy.get_param("/rl_client/actor_checkpoint")
    critic_checkpoint = rospy.get_param("/rl_client/critic_checkpoint")
    fig_path = rospy.get_param("/rl_client/train_fig_path")
    result_path = rospy.get_param("/rl_client/result_path")
    test_result_path = rospy.get_param("/rl_client/test_result_path")
    

    #init
    obs_dim = 7
    action_dim = 6
    rospy.init_node('RL_client', anonymous=True)
    sess = tf.compat.v1.Session()
    K.set_session(sess)
    actor_critic = ActorCritic(sess)

    num_trials = 50000
    trial_len  = rospy.get_param("/rl_client/num_steps")
    #trial_len  = 75
    #env = gym.make('Pendulum-v0')

    for i in range(num_trials):

        reset()
        state = np.zeros((1,obs_dim))
        #state = env.reset()
        reward_sum = 0
        done = False
        for j in range(trial_len):
            state = state.reshape((1,obs_dim))
            #env.render()
            print("trial:" + str(i))
            print("step:" + str(j))
            print(state.shape)
            raw_action, action = actor_critic.act(state,0)
            #action = actor_critic.actor_model.predict([state]) + np.random.normal(size=(action_dim,))
            action = action.reshape((action_dim))
            last=0
            if j == (trial_len - 1):
                last = 1
            next_state, reward = step(action,last)
            #next_state, reward, done, _ = env.step(action)
            #TODO: Add time step
            next_state = list(next_state)
            next_state.append(float(j+1)/5)
            next_state = np.asarray(next_state)
            reward_sum += reward
            rospy.loginfo(rospy.get_caller_id() + 'got reward %s',reward)
            if j == (trial_len - 1):
                done = True


            #train the agent
            if len(actor_critic.memory) > actor_critic.batch_size and done:
                critic_loss, actor_loss = actor_critic.train()
                actor_critic.update_target()
                critic_loss_list.append(critic_loss)
                actor_loss_list.append(actor_loss)
                wandb.log({"critic_loss": critic_loss})
                wandb.log({"actor_loss": actor_loss})

            #record data
            raw_action = raw_action.reshape((1,action_dim))
            actor_critic.remember(state, raw_action, reward, next_state, done)

            state = next_state

            if done:
                rospy.loginfo(rospy.get_caller_id() + 'got total reward %s',reward_sum)
                reward_list.append(reward_sum)
                wandb.log({"reward_sum": reward_sum}) 
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
            state = np.zeros((obs_dim,))
            #state = env.reset()
            test_reward_sum = 0
            test_reward_sum = 0
            done = False
            for j in range(trial_len):
                state = state.reshape((1,obs_dim))
                #env.render()
                print("test:" + str(i))
                print("step:" + str(j))

                raw_action, action = actor_critic.act(state,1)
                #action = actor_critic.actor_model.predict([state])
                action = action.reshape((action_dim))
                last=0
                if j == (trial_len - 1):
                    last = 1
                state, reward = step(action,last)
                #TODO: Add time step
                state = list(state)
                state.append(float(j+1)/5)
                #state, reward, done, _ = env.step(action)
                state = np.asarray(state)
                test_reward_sum += reward
                rospy.loginfo(rospy.get_caller_id() + 'got reward %s',reward)
                if j == (trial_len - 1):
                    done = True

                
                if done:
                    rospy.loginfo(rospy.get_caller_id() + 'got total reward %s',test_reward_sum)
                    test_reward_list.append(test_reward_sum)
                    fig1, ax1 = plt.subplots()
                    ax1.plot(test_reward_list)
                    test_fig_path = rospy.get_param("/rl_client/test_fig_path")
                    fig1.savefig(test_fig_path)
                    np.savetxt(test_result_path, test_reward_list, fmt='%f')
                    wandb.log({"test_reward_sum": test_reward_sum})
                    break



if __name__ == '__main__':
    simulation()
