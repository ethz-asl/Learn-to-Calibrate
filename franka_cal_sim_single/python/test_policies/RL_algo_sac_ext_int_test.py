import rospy
from franka_cal_sim_single.srv import*


import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


import wandb




class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action



class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z      = normal.sample()
        action = torch.tanh(z)
        
        action  = action.detach().cpu().numpy()
        return action[0]




def act(cur_state, rand):

    if rand==1:
        raw_action = random.uniform(-1.*np.ones(6), 1.*np.ones(6))
    else:
        #network out put:
        raw_action = policy_net.get_action(cur_state)
    #scale of actions
    scale = np.asarray([0.3,1.57,1.57,0.4,0.4,0.4]) #0.4 0.5 0.6 0.8
    det_action = np.multiply(raw_action,scale)


    return raw_action, det_action



def soft_q_update(batch_size, 
           gamma=0.95,
           mean_lambda=1e-3,
           std_lambda=1e-3,
           z_lambda=0.0,
           soft_tau=1e-2,
          ):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    expected_q_value = soft_q_net(state, action)
    expected_value   = value_net(state)
    new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)


    target_value = target_value_net(next_state)
    next_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())

    expected_new_q_value = soft_q_net(state, new_action)
    next_value = expected_new_q_value - log_prob
    value_loss = value_criterion(expected_value, next_value.detach())

    log_prob_target = expected_new_q_value - expected_value
    policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
    

    mean_loss = mean_lambda * mean.pow(2).mean()
    std_loss  = std_lambda  * log_std.pow(2).mean()
    z_loss    = z_lambda    * z.pow(2).sum(1).mean()

    policy_loss += mean_loss + std_loss + z_loss

    soft_q_optimizer.zero_grad()
    q_value_loss.backward()
    soft_q_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


    wandb.log({"value_loss": value_loss})
    wandb.log({"q_value_loss": q_value_loss})
    wandb.log({"actor_loss": policy_loss})




def step(action, last):
    rospy.wait_for_service('/model_client/rl_service')
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    take_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
    
    try:
        resp1 = take_step(0, action.tolist(), last, 0)
    except:
        return np.zeros((18,)), 0, 0
    return resp1.next_state, resp1.reward, resp1.done

def reset(restart):
    rospy.wait_for_service('/model_client/rl_service')
    rospy.loginfo(rospy.get_caller_id() + 'begin service')
    reset_step = rospy.ServiceProxy('/model_client/rl_service', RLSrv)
    try:
        resp1 = reset_step(1, [], 0, restart)
    except:
        return 0
    return resp1.reward


actor_checkpoint = rospy.get_param("/rl_client/test_actor_checkpoint")
critic_checkpoint = rospy.get_param("/rl_client/test_critic_checkpoint")


action_dim = 6
state_dim  = 24
hidden_dim = 256
rospy.init_node('RL_client', anonymous=True)

value_net        = ValueNetwork(state_dim, hidden_dim).to(device)
target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    

value_criterion  = nn.MSELoss()
soft_q_criterion = nn.MSELoss()

value_lr  = 3e-4
soft_q_lr = 3e-4
policy_lr = 1e-4

value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer = optim.Adam(soft_q_net.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)


replay_buffer_size = 1000000
replay_buffer = ReplayBuffer(replay_buffer_size)


max_frames  = 500000
max_steps   = rospy.get_param("/rl_client/num_steps")
frame_idx   = 0
rewards     = []
batch_size  = 128

max_frames  = 50

reward_list = []
episode = 0


#for test:
policy_net.load_state_dict(torch.load(actor_checkpoint))
data_list = []
ave_translation = 0
ave_rotation = 0

while episode < 4: 
    print("episode: " + str(episode))
    reset(0)
    state = np.concatenate([np.zeros((6, )), np.ones((12, )) * 0.5, np.zeros((6, ))])
    #init size
    state[8] = 0.28
    state[9] = 0.28
    episode_reward = 0
    last = 0
    done = 0
    episode += 1

    j = 0

    #record experiment data
    '''distortion (3), dist_center(2), intrinsic accuracy, coverage, info_gain, path length, episode length, rewards'''
    
    #intrinsic, extrinsic
    distortion_K = rospy.get_param('rl_client/K')
    actions = []
    dist_center = rospy.get_param('rl_client/center')
    if_test_random = rospy.get_param('rl_client/if_test_random')
    for j in range(max_steps):

        # ablation study
        ablation = rospy.get_param('rl_client/ablation')
        print(ablation)

        if ablation == 'visual': # remove visual
            state[6:18] = 0.5
        if ablation == 2: # remove motion
            state[0:6] = 0.5
        
        #execute the action
        if if_test_random:
            raw_action, action = act(state, 1)
        else:
            raw_action, action = act(state,0)
        
        
        next_state, reward, done = step(action, 0)


        next_state = np.asarray(next_state)
        actions.append(action)
        print(next_state.shape)

        
        rospy.loginfo(rospy.get_caller_id() + 'got state %s',state)
        
        state = next_state
        episode_reward += reward
        frame_idx += 1

        rospy.loginfo(rospy.get_caller_id() + 'got reward %s',reward)
        rospy.loginfo(rospy.get_caller_id() + 'got action %s',action)
        
        actions = np.asarray(actions)
        np.savetxt("new_test_actions_real.txt", actions)
        actions = actions.tolist()

        
        if done or j==max_steps-1:
            print("total_rewards: "+str(episode_reward))

            

            #record the experiment data
            cal_err = rospy.get_param('rl_client/calibration_err')
            reproj_err = rospy.get_param('rl_client/reproj_err')
            coverage = np.sum(np.abs(next_state[:8]-0.5)+next_state[9]-next_state[8]+next_state[11]-next_state[10])
            info_gain = rospy.get_param("rl_client/a_opt")
            path_len = rospy.get_param("rl_client/translation_and_rotation")
            translation = rospy.get_param("rl_client/total_translation")
            rotation = rospy.get_param("rl_client/total_rotation")
            total_time = rospy.get_param("rl_client/cal_time")
            episode_len = j+1

            data = distortion_K + dist_center
            data.append(reproj_err)
            data.append(cal_err)
            data.append(coverage)
            data.append(info_gain)
            data.append(float(path_len))
            data.append(float(episode_len))
            data.append(float(episode_reward))
            data.append(float(translation))
            data.append(float(rotation))
            data.append(float(total_time))
            ave_translation += translation / 10
            ave_rotation += rotation / 10
            

            break

        
    rewards.append(episode_reward)
    if episode>1:
        data_list.append(np.asarray(data))

data_array = np.asarray(data_list)
print(data_array[:, -2])
print(data_array[:, -1])
print(ave_translation)
print(ave_rotation)
np.savetxt("experiment_data",data_array)
