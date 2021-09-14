import rospy
from franka_cal_sim_single.srv import*
from SimplifiedCalEnv import CalEnv, DoubleCalEnv, DoubleCalImuEnv, DoubleCalImuSetEnv


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


wandb.init(project="semester-project")
wandb.config["more"] = "custom"



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, set_state, nom_state, action, reward, next_set_state, next_nom_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (set_state, nom_state, action, reward, next_set_state, next_nom_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        set_state, nom_state, action, reward, next_set_state, next_nom_state, done = map(np.stack, zip(*batch))
        return set_state, nom_state, action, reward, next_set_state, next_nom_state, done
    
    def __len__(self):
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
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
    def __init__(self, set_state_dim, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.conv1 = nn.Conv1d(set_state_dim,hidden_dim,1)
        self.conv2 = nn.Conv1d(hidden_dim,hidden_dim,1)
        self.conv3 = nn.Conv1d(hidden_dim,hidden_dim,1)
        self.conv4 = nn.Conv1d(2*hidden_dim,hidden_dim,1)
        self.conv5 = nn.Conv1d(hidden_dim,hidden_dim,1)
        self.conv6 = nn.Conv1d(hidden_dim,hidden_dim,1)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, set_state, nom_state):
        set_x = F.sigmoid(self.conv1(set_state))
        set_x = F.sigmoid(self.conv2(set_x))
        set_x = F.sigmoid(self.conv3(set_x))
        pool = nn.MaxPool1d(set_x.size(-1))(set_x)
        flat1 = nn.Flatten(1)(pool)

        set_x = torch.cat((set_x, pool.repeat(1, 1, set_x.size(-1))), dim=-2)
        set_x = F.sigmoid(self.conv4(set_x))
        set_x = F.sigmoid(self.conv5(set_x))
        set_x = F.sigmoid(self.conv6(set_x))
        pool = nn.MaxPool1d(set_x.size(-1))(set_x)
        flat = nn.Flatten(1)(pool)
        # flat1 = nn.Flatten(1)(ave_pool)
        # min_pool = torch.min(set_x,dim=-1)

        x = F.relu(self.linear1(nom_state))
        x = F.relu(self.linear2(torch.cat((x, flat), dim=-1)))
        x = self.linear3(x)
        x = self.linear4(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, set_num_inputs, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.conv1 = nn.Conv1d(set_num_inputs,hidden_size,1)
        self.conv2 = nn.Conv1d(hidden_size,hidden_size,1)
        self.conv3 = nn.Conv1d(hidden_size,hidden_size,1)
        self.conv4 = nn.Conv1d(2*hidden_dim,hidden_dim,1)
        self.conv5 = nn.Conv1d(hidden_dim,hidden_dim,1)
        self.conv6 = nn.Conv1d(hidden_dim,hidden_dim,1)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(2*hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, 1)
        
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, set_state, nom_state, action):
        set_x = F.sigmoid(self.conv1(set_state))
        set_x = F.sigmoid(self.conv2(set_x))
        set_x = F.sigmoid(self.conv3(set_x))
        pool = nn.MaxPool1d(set_x.size(-1))(set_x)
        flat1 = nn.Flatten(1)(pool)

        set_x = torch.cat((set_x, pool.repeat(1, 1, set_x.size(-1))), dim=-2)
        set_x = F.sigmoid(self.conv4(set_x))
        set_x = F.sigmoid(self.conv5(set_x))
        set_x = F.sigmoid(self.conv6(set_x))
        pool = nn.MaxPool1d(set_x.size(-1))(set_x)
        ave_pool = nn.AvgPool1d(set_x.size(-1))(set_x)
        flat = nn.Flatten(1)(pool)
        # flat1 = nn.Flatten(1)(ave_pool)
        # min_pool = torch.min(set_x,dim=-1)

        x = torch.cat([nom_state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(torch.cat((x, flat), dim=-1)))
        x = self.linear3(x)
        x = self.linear4(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, set_num_inputs, nom_num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(nom_num_inputs, hidden_size)
        self.linear2 = nn.Linear(2*hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)

        self.conv1 = nn.Conv1d(set_num_inputs,hidden_size,1)
        self.conv2 = nn.Conv1d(hidden_size,hidden_size,1)
        self.conv3 = nn.Conv1d(hidden_size,hidden_size,1)
        self.conv4 = nn.Conv1d(2*hidden_dim,hidden_dim,1)
        self.conv5 = nn.Conv1d(hidden_dim,hidden_dim,1)
        self.conv6 = nn.Conv1d(hidden_dim,hidden_dim,1)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, set_state, nom_state):
        set_x = F.sigmoid(self.conv1(set_state))
        set_x = F.sigmoid(self.conv2(set_x))
        set_x = F.sigmoid(self.conv3(set_x))
        pool = nn.MaxPool1d(set_x.size(-1))(set_x)
        flat1 = nn.Flatten(1)(pool)

        set_x = torch.cat((set_x, pool.repeat(1, 1, set_x.size(-1))), dim=-2)
        set_x = F.sigmoid(self.conv4(set_x))
        set_x = F.sigmoid(self.conv5(set_x))
        set_x = F.sigmoid(self.conv6(set_x))
        pool = nn.MaxPool1d(set_x.size(-1))(set_x)
        flat = nn.Flatten(1)(pool)
        # flat1 = nn.Flatten(1)(ave_pool)
        # min_pool = torch.min(set_x,dim=-1)

        x = F.relu(self.linear1(nom_state))
        x = F.relu(self.linear2(torch.cat((x, flat), dim=-1)))
        x = F.relu(self.linear3(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, 
    
    def evaluate(self, set_state, nom_state, epsilon=1e-6):
        mean, log_std = self.forward(set_state, nom_state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, set_state, nom_state):
        set_state = torch.FloatTensor(set_state).unsqueeze(0).to(device)
        nom_state = torch.FloatTensor(nom_state).unsqueeze(0).to(device)
        mean, log_std = self.forward(set_state, nom_state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z      = normal.sample()
        action = torch.tanh(z)
        
        action  = action.detach().cpu().numpy()
        return action[0]




def act(set_state, nom_state):
    #network out put:
    raw_action = policy_net.get_action(set_state, nom_state)
    #scale of actions
    scale = np.asarray([0.3,1.57,1.57,0.4,0.4,0.4])
    det_action = np.multiply(raw_action,scale)


    return raw_action, det_action



def soft_q_update(batch_size, 
           gamma=0.9,
           mean_lambda=1e-3,
           std_lambda=1e-3,
           z_lambda=0.0,
           soft_tau=1e-2,
          ):
    set_state, nom_state, action, reward, next_set_state, next_nom_state, done = replay_buffer.sample(batch_size)
    #print(set_state.shape)
    nom_state = torch.FloatTensor(nom_state).to(device)
    set_state  = torch.FloatTensor(set_state).to(device)
    next_set_state = torch.FloatTensor(next_set_state).to(device)
    next_nom_state = torch.FloatTensor(next_nom_state).to(device)
    action     = torch.FloatTensor(action).to(device)
    reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    expected_q_value = soft_q_net(set_state, nom_state, action)
    expected_value   = value_net(set_state, nom_state)
    new_action, log_prob, z, mean, log_std = policy_net.evaluate(set_state, nom_state)


    target_value = target_value_net(next_set_state, next_nom_state)
    next_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())

    expected_new_q_value = soft_q_net(set_state, nom_state, new_action)
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





env = NormalizedActions(DoubleCalImuSetEnv())

action_dim = env.action_space.shape[0]
set_obs_dim  = env.set_obs_dim
nom_obs_dim = env.nom_obs_dim
hidden_dim = 256


value_net        = ValueNetwork(set_obs_dim, nom_obs_dim, hidden_dim).to(device)
target_value_net = ValueNetwork(set_obs_dim, nom_obs_dim, hidden_dim).to(device)

soft_q_net = SoftQNetwork(set_obs_dim, nom_obs_dim, action_dim, hidden_dim).to(device)
policy_net = PolicyNetwork(set_obs_dim, nom_obs_dim, action_dim, hidden_dim).to(device)
print(policy_net)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)
    

value_criterion  = nn.MSELoss()
soft_q_criterion = nn.MSELoss()

value_lr  = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr, weight_decay=0)
soft_q_optimizer = optim.Adam(soft_q_net.parameters(), lr=soft_q_lr, weight_decay=0)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr, weight_decay=0)


replay_buffer_size = 100000
replay_buffer = ReplayBuffer(replay_buffer_size)


max_frames  = 400000
max_steps   = 10
frame_idx   = 0
rewards     = []
batch_size  = 128

max_frames  = 1000000

#save path:
actor_checkpoint = "/home/yunke/prl_proj/panda_ws/src/franka_cal_sim_single/python/checkpoints/actor_checkpoint"
critic_checkpoint = "/home/yunke/prl_proj/panda_ws/src/franka_cal_sim_single/python/checkpoints/critic_checkpoint"


reward_list = []
episode = 0
while frame_idx < max_frames:
    set_state, nom_state = env.reset()
    episode_reward = 0
    for step in range(max_steps):
        # save model and visualize
        if episode%200==0:
            torch.save(value_net.state_dict(), critic_checkpoint)
            torch.save(policy_net.state_dict(), actor_checkpoint)
            env.visualize()
            
        if episode % 1000 == 0 and episode < 5000:
            value_lr *= 0.7
            soft_q_lr *= 0.7
            policy_lr *= 0.7

        action = policy_net.get_action(set_state, nom_state)
        next_set_state, next_nom_state, reward, done, info = env.step(action)
        
        replay_buffer.push(set_state, nom_state, action, reward, next_set_state, next_nom_state, done)
        if len(replay_buffer) > batch_size:
            soft_q_update(batch_size)
        
        set_state = next_set_state
        nom_state = next_nom_state
        episode_reward += reward
        frame_idx += 1
        # if step == (max_steps - 1):
        #     done = True
        if episode%200==0:
            env.visualize()
            print(info)
        
        
        if done or step == max_steps-1:
            print("episode: "+str(episode))
            print("length: "+str(step+1))
            wandb.log({"total_rewards": episode_reward})
            wandb.log({"episode length": step+1})
            print("total_rewards: "+str(episode_reward))
            if episode%10==0:
                wandb.log({"ave_rewards": sum(rewards[-10:])/10})
            break
            
    episode+=1

       

            
        
    rewards.append(episode_reward)
