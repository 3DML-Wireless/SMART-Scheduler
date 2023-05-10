import random
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# from dqn_agent import Agent
from itertools import combinations
import h5py
from mimo_sim_ul import *
from usr_group import *
from torch.utils.tensorboard import SummaryWriter

"""# Build DQN Agent and Helper Functions"""

H_file = h5py.File('./dataset/4_2_4_mob_normal.hdf5','r')
H_r = np.array(H_file.get('H_r'))
H_i = np.array(H_file.get('H_i'))
H = np.array(H_r + 1j*H_i)
print("H_1 shape is:", H.shape)

se_max_ur = np.array(H_file.get('se_max'))
print("se_max_ur_1 shape is:", se_max_ur.shape)

normal_ur = np.array(H_file.get('normal'))
print("normal_ur_1 shape is:", normal_ur.shape)

sel_ue = 2
num_ue = 4
num_bs = 4
user_set = [0,1,2,3]


a = 1
b = 1
c = 0
reward_scale = 10

writer = SummaryWriter('runs/{}_Double_DQN'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

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

"""#### Check that there is a GPU avaiable"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

"""# Start Environment and Build Dueling DQN Agent"""

num_state_feats = 32
num_actions = 10
# max_observation_values = env.observation_space.high
# print('Number of state features: {}'.format(num_state_feats))
# print('Number of possible actions: {}'.format(num_actions))

class DoubleDQN(nn.Module):
  """Convolutional neural network for the Atari games."""
  def __init__(self, num_inputs, num_actions):
        super(DoubleDQN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, num_actions)

  def forward(self, states):
      """Forward pass."""
      x = F.relu(self.fc1(states))
      x = F.relu(self.fc2(x))
      return self.out(x)

    
# Create main and target neural networks.
main_nn = DoubleDQN(num_state_feats, num_actions).to(device)
target_nn = DoubleDQN(num_state_feats, num_actions).to(device)

# Loss function and optimizer.
optimizer = torch.optim.Adam(main_nn.parameters(), lr=1e-3)
loss_fn = nn.SmoothL1Loss()  # Huber loss

"""# Create Helper Functions"""

def select_epsilon_greedy_action(state, epsilon):
  """Take random action with probability epsilon, else take best action."""
  result = np.random.uniform()
  if result < epsilon:
    return random.randint(0,num_actions-1) # Random action.
  else:
    qs = main_nn(state).cpu().data.numpy()
    return np.argmax(qs) # Greedy action for state.

class UniformBuffer(object):
  """Experience replay buffer that samples uniformly."""

  def __init__(self, size, device):
    self._size = size
    self.buffer = []
    self.device = device
    self._next_idx = 0

  def add(self, state, action, reward, next_state, done):
    if self._next_idx >= len(self.buffer):
      self.buffer.append((state, action, reward, next_state, done))
    else:
      self.buffer[self._next_idx] = (state, action, reward, next_state, done)
    self._next_idx = (self._next_idx + 1) % self._size

  def __len__(self):
    return len(self.buffer)

  def sample(self, num_samples):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    idx = np.random.choice(len(self.buffer), num_samples)
    for i in idx:
      elem = self.buffer[i]
      state, action, reward, next_state, done = elem
      states.append(np.array(state, copy=False))
      actions.append(np.array(action, copy=False))
      rewards.append(reward)
      next_states.append(np.array(next_state, copy=False))
      dones.append(done)
    states = torch.as_tensor(np.array(states), device=self.device)
    actions = torch.as_tensor(np.array(actions), device=self.device)
    rewards = torch.as_tensor(np.array(rewards, dtype=np.float32),
                              device=self.device)
    next_states = torch.as_tensor(np.array(next_states), device=self.device)
    dones = torch.as_tensor(np.array(dones, dtype=np.float32),
                            device=self.device)
    return states, actions, rewards, next_states, dones

"""# Set Up Function to Perform a Training Step"""

def train_step(states, actions, rewards, next_states, dones):
  """Perform a training iteration on a batch of data."""
  next_qs_argmax = main_nn(next_states).argmax(dim=-1, keepdim=True)
  masked_next_qs = target_nn(next_states).gather(1, next_qs_argmax).squeeze()
  target = rewards + (1.0 - dones) * discount * masked_next_qs
  # masked_qs = main_nn(states).gather(1, actions.unsqueeze(dim=-1)).squeeze()
  masked_qs = main_nn(states).gather(1, actions).squeeze()
  loss = loss_fn(masked_qs, target.detach())
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss

"""# Start running the DQN algorithm and see how the algorithm learns."""

# Hyperparameters.
num_episodes = 1000 # @param {type:"integer"}
epsilon = 1.0 # @param {type:"number"}
batch_size = 32 # @param {type:"integer"}
discount = 0.99 # @param {type:"number"}
buffer_size = 200000 # @param {type:"integer"}
updates = 0
buffer = UniformBuffer(size=buffer_size, device=device)
num_iters = 400
# Start training. Play game once and then train with a batch.
last_100_ep_rewards, cur_frame = [], 0
for episode in range(num_episodes+1):
  state = np.concatenate((np.reshape(H_r[0,:,:],(1,-1)),np.reshape(H_i[0,:,:],(1,-1))),axis = 1)
  # state = np.reshape(state,(4,4,2))
  ep_reward, done = 0, False
  ue_history = 0.01*np.ones((num_ue,))
  episode_steps = 0
  while not done:
    state_np = np.array(state, dtype=np.float32)
    state_in = torch.as_tensor(np.expand_dims(state_np, axis=0),
                               device=device)
    action = select_epsilon_greedy_action(state_in, epsilon)
    ue_select,idx = sel_ue(action)

    mod_select = np.ones((idx,)) *16
    ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(H[episode_steps,:,ue_select],(num_bs,-1)),idx,mod_select)
    ur_se_total = ur_se_total/normal_ur[episode_steps]
    for i in range(0,idx):
        ue_history[ue_select[i]] += ur_se[i]

    jfi = np.square((np.sum(ue_history))) / (num_ue * np.sum(np.square(ue_history)))
    if episode_steps == num_iters -2:
        print('jfi is', episode_steps, jfi)

    next_state = np.concatenate((np.reshape(H_r[(episode_steps+1),:,:],(1,-1)),np.reshape(H_i[(episode_steps+1),:,:],(1,-1))),axis = 1)
    next_state = next_state.astype(np.float32)
    # next_state = np.reshape(next_state,(4,4,2))
    reward  = (a*ur_se_total + b*jfi)*reward_scale
    done = False
    # print('reward is:', reward)
    if num_iters and episode_steps >= num_iters-2:
        done = True

    # next_state, reward, done, info = env.step(action)
    # next_state = next_state.astype(np.float32)
    ep_reward += reward
    # Save to experience replay.
    buffer.add(state, action, reward, next_state, done)
    state = next_state
    episode_steps += 1
    cur_frame += 1
    if epsilon > 0.01:
      epsilon -= 1.1e-6

    if len(buffer) >= batch_size:
      states, actions, rewards, next_states, dones = buffer.sample(batch_size)
      states = states.type(torch.FloatTensor).to(device)
      next_states = next_states.type(torch.FloatTensor).to(device)
      loss = train_step(states, actions, rewards, next_states, dones)
      updates += 1
      writer.add_scalar('loss', loss, updates)

    # Copy main_nn weights to target_nn.
    if cur_frame % 10000 == 0:
      target_nn.load_state_dict(main_nn.state_dict())
  
  writer.add_scalar('reward', ep_reward, episode)

  if len(last_100_ep_rewards) == 100:
    last_100_ep_rewards = last_100_ep_rewards[1:]
  last_100_ep_rewards.append(ep_reward)

  if episode % 25 == 0:
    print(f'Episode: {episode}/{num_episodes}, Epsilon: {epsilon:.3f}, '\
          f'Loss: {loss:.4f}, Return: {np.mean(last_100_ep_rewards):.2f}')

