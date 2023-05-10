import torch
import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import numpy as np
# import gym
import random
from collections import namedtuple, deque

from itertools import combinations
import h5py
from mimo_sim_ul import *
from usr_group import *
from torch.utils.tensorboard import SummaryWriter


### Import data from hdf5 file #############################

H_file = h5py.File('./dataset/4_2_4_mob_normal.hdf5','r')
H_r = np.array(H_file.get('H_r'))
H_i = np.array(H_file.get('H_i'))
# H_i = np.transpose(H_i,(2,1,0))
# H_r = np.transpose(H_r,(2,1,0))
# print("H_r_1 shape is:", H_r_1.shape)
# print("H_i_1 shape is:", H_i_1.shape)
H = np.array(H_r + 1j*H_i)
print("H_1 shape is:", H.shape)

se_max_ur = np.array(H_file.get('se_max'))
print("se_max_ur_1 shape is:", se_max_ur.shape)

normal_ur = np.array(H_file.get('normal'))
print("normal_ur_1 shape is:", normal_ur.shape)



#Tensorboard
writer = SummaryWriter('runs/{}_Double_DQN'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

sel_ue = 2
num_ue = 4
num_bs = 4
randoms = 1.
epsilon = 80000

user_set = [0,1,2,3]
a = 1
b = 1
c = 0
reward_scale = 1
### Record vector defination
reward_record = np.zeros((800,))
history_record = np.zeros((10,400,num_ue))
# max_history_record = np.zeros((400,num_ue))
# ue_history = 0.01*np.ones((num_ue,))


def sel_ue(action):
    sum_before = 0
    for i in range (1,3):
        sum_before += len(list(combinations(user_set, i)))
        if ((action+1)>sum_before):
            continue
        else:
            idx = i
            sum_before -= len(list(combinations(user_set, i)))
            ue_select = list(combinations(user_set, i))[action-sum_before]
            break
    return ue_select,idx

class BasicBuffer:

  def __init__(self, max_size):
      self.max_size = max_size
      self.buffer = deque(maxlen=max_size)
      self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

  def push(self, state, action, reward, next_state, done):
      e = self.experience(state, action, np.array([reward]), next_state, done)
      self.buffer.append(e)

  def sample(self, batch_size):
    #   state_batch = []
    #   action_batch = []
    #   reward_batch = []
    #   next_state_batch = []
    #   done_batch = []

      experiences = random.sample(self.buffer, batch_size)

    #   for experience in batch:
    #       state, action, reward, next_state, done = experience
    #       state_batch.append(state)
    #       action_batch.append(action)
    #       reward_batch.append(reward)
    #       next_state_batch.append(next_state)
    #       done_batch.append(done)

      states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
      actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None]))
      rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
      next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
      dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()
      
      return (states, actions, rewards, next_states, dones)
    #   return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

  def __len__(self):
      return len(self.buffer)


def mini_batch_train(agent, max_episodes, max_steps, batch_size, randoms, epsilon):
    episode_rewards = []
    updates = 0
    eps = randoms
    # eps = eps_start
    for i_episode in range(max_episodes):
        episode_reward = 0
        # group_idx = usr_group(np.squeeze(H[0,:,:]))
        done = False
        episode_steps = 0
        ue_ep_history = np.zeros((400,num_ue))
        ue_history = 0.01*np.ones((num_ue,))

        # state = np.concatenate((np.reshape(se_max_ur[0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(group_idx,(1,-1))),axis = 1)
        state = np.reshape(se_max_ur[0,:],(1,num_ue))
        while not done:
            start_time = time.time()
            action = agent.get_action(state, eps)
            # next_state, reward, done, _ = env.step(action)
            ue_select,idx = sel_ue(action)

            mod_select = np.ones((idx,)) *16
            ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(H[episode_steps,:,ue_select],(num_bs,-1)),idx,mod_select)
            # ur_se_total = ur_se_total/normal_ur[episode_steps]
            eps -= 1/epsilon
            if eps < 0:
                eps = 0
            # print('se_total:',ur_se_total)
            # print('idx is',idx)
            # print('ue_selection is:', ue_select)
            
            for i in range(0,idx):
                ue_history[ue_select[i]] += ur_se[i]
            ue_ep_history[episode_steps,:] = ue_history
            if (i_episode >= 790):
                history_record[i_episode-790,episode_steps,:] = ue_history

            jfi = np.square((np.sum(ue_history))) / (num_ue * np.sum(np.square(ue_history)))
            if episode_steps == max_steps-2:
                print('jfi is', jfi)
            # group_idx_next = usr_group(np.squeeze(H[(episode_steps+1),:,:]))
            # next_state = np.concatenate((np.reshape(se_max_ur[(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(group_idx_next,(1,-1))),axis = 1)
            next_state = np.reshape(se_max_ur[(episode_steps+1),:],(1,num_ue))
            reward  = (a*ur_se_total)*reward_scale
            done = False

            if max_steps and episode_steps >= max_steps-2:
                done = True

            agent.replay_buffer.push(state, action, reward, next_state, done)

            if len(agent.replay_buffer) > batch_size:
                updates += 1
                agent.update(batch_size, updates)   

            # if done or step == max_steps-1:
            #     episode_rewards.append(episode_reward)
            #     print("Episode " + str(i_episode) + ": " + str(episode_reward))
            #     break

            state = next_state
            episode_reward += reward
            

            episode_steps += 1
            updates += 1
            end_time = time.time()

        writer.add_scalar('reward', episode_reward, i_episode)
        reward_record[i_episode] = episode_reward
        # scores_window.append(score)           # save most recent score
        episode_rewards.append(episode_reward)  # save most recent score
        # eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(episode_rewards)),'\n')
        if (i_episode > 500) and (i_episode % 10 == 0):
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(episode_rewards)),'\n')
        # if np.mean(scores_window)>=200.0:
        #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.model.state_dict(), 'checkpoint_ddqn.pth')

    return episode_rewards


# class ConvDQN(nn.Module):
    
#     def __init__(self, input_dim, output_dim):
#         super(ConvDQN, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.fc_input_dim = self.feature_size()
        
#         self.conv = nn.Sequential(
#             nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1),
#             nn.ReLU()
#         )

#         self.fc = nn.Sequential(
#             nn.Linear(self.fc_input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, self.output_dim)
#         )

#     def forward(self, state):
#         features = self.conv_net(state)
#         features = features.view(features.size(0), -1)
#         qvals = self.fc(features)
#         return qvals

#     def feature_size(self):
#         return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


class DQN(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals


class  Agent:

    def __init__(self, state_size, action_size, use_conv=False, learning_rate=3e-3, gamma=0.99, tau=0.01, buffer_size=100000):
        # self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
	
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.use_conv = use_conv
        if self.use_conv:
            self.model = ConvDQN(state_size, action_size).to(self.device)
            self.target_model = ConvDQN(state_size, action_size,).to(self.device)
        else:
            self.model = DQN(state_size, action_size).to(self.device)
            self.target_model = DQN(state_size, action_size,).to(self.device)
        
        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        
    def get_action(self, state, eps=0.):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        
        if(np.random.random() < eps):
            # print("random action:\n")
            return random.choice(np.arange(self.action_size))
        # print("Max action:", action,"\n")
        return action

    def compute_loss(self, batch):     
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # resize tensors
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)

        # compute loss
        # print('states size and actions size:',states.size(), actions.size())
        curr_Q = self.model.forward(states).gather(1, actions)
        next_Q = self.target_model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q
        
        loss = F.mse_loss(curr_Q, expected_Q.detach())
        
        return loss

    def update(self, batch_size, updates):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)
        # print("loss is:", loss)
        writer.add_scalar('loss', loss, updates)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # target network update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


# env_id = "CartPole-v0"
MAX_EPISODES = 800
MAX_STEPS = 400
BATCH_SIZE = 128

# env = gym.make(env_id)
agent = DQNAgent(state_size=4, action_size=10, use_conv=False)
episode_rewards = mini_batch_train(agent, MAX_EPISODES, MAX_STEPS, BATCH_SIZE, randoms, epsilon)

with h5py.File("./results/4_2_4_DoubleDQN_mobile.hdf5", "w") as data_file:
    data_file.create_dataset("reward", data=reward_record)
    data_file.create_dataset("history", data=history_record)
    # data_file.create_dataset("max_history", data=max_history_record)

print('Training is finished\n')