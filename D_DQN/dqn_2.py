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

"""# 4 UEs"""
H_file = h5py.File('./dataset/4_2_4_mob_normal.hdf5','r')
H_r = np.array(H_file.get('H_r'))
H_i = np.array(H_file.get('H_i'))
H = np.array(H_r + 1j*H_i)
print("H shape is:", H.shape)

se_max_ur = np.array(H_file.get('se_max'))
print("se_max_ur shape is:", se_max_ur.shape)

normal_ur = np.array(H_file.get('normal'))
print("normal_ur shape is:", normal_ur.shape)

num_features = 4
num_actions = 10

sel_ue = 2
num_ue = 4
num_bs = 4
user_set = [0,1,2,3]

"""# 8 UEs"""

# H_file = h5py.File('./dataset/8_8_2RB_mob_normal.hdf5','r')
# H_r = np.array(H_file.get('H_r_1'))
# H_i = np.array(H_file.get('H_i_1'))

# H = np.array(H_r + 1j*H_i)
# print("H shape is:", H.shape)

# se_max_ur = np.array(H_file.get('se_max_1'))
# print("se_max_ur shape is:", se_max_ur.shape)

# normal_ur = np.array(H_file.get('normal_1'))
# print("normal_ur shape is:", normal_ur.shape)

# num_features = 8
# num_actions = 162

# sel_ue = 4
# num_ue = 8
# num_bs = 8
# user_set = [0,1,2,3,4,5,6,7]


a = 1
b = 1
c = 0
reward_scale = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter('runs/{}_DQN'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

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

class DQN(nn.Module):
    """Dense neural network class."""
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, num_actions)

    def forward(self, states):
        """Forward pass."""
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        return self.out(x)

main_nn = DQN(num_features, num_actions).to(device)
target_nn = DQN(num_features, num_actions).to(device)

optimizer = torch.optim.Adam(main_nn.parameters(), lr=3e-3)
loss_fn = nn.MSELoss()

class ReplayBuffer(object):
  """Experience replay buffer that samples uniformly."""
  def __init__(self, size, device="cpu"):
    """Initializes the buffer."""
    self.buffer = deque(maxlen=size)
    self.device = device

  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

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
    rewards = torch.as_tensor(
        np.array(rewards, dtype=np.float32), device=self.device
    )
    next_states = torch.as_tensor(np.array(next_states), device=self.device)
    dones = torch.as_tensor(np.array(dones, dtype=np.float32), device=self.device)
    return states, actions, rewards, next_states, dones

def select_epsilon_greedy_action(state, epsilon):
  """Take random action with probability epsilon, else take best action."""
  result = np.random.uniform()
  if result < epsilon:
    # print("Random Actions")
    return random.randint(0,num_actions-1) # Random action (left or right).
  else:
    qs = main_nn(state).cpu().data.numpy()
    # print("Learned Actions")
    return np.argmax(qs) # Greedy action for state.

def train_step(states, actions, rewards, next_states, dones):
  """Perform a training iteration on a batch of data sampled from the experience
  replay buffer.
  """
  # Calculate targets.
  max_next_qs = target_nn(next_states).max(-1).values
  target = rewards + (1.0 - dones) * discount * max_next_qs
  qs = main_nn(states)
  action_masks = F.one_hot(actions, num_actions)
  masked_qs = (action_masks * qs).sum(dim=-1)
  loss = loss_fn(masked_qs, target.detach())
  #nn.utils.clip_grad_norm_(loss, max_norm=10)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss

"""# Start running the DQN algorithm and see how the algorithm learns."""

# Hyperparameters.
num_episodes = 1000
epsilon = 1.0
batch_size = 32
discount = 0.99
buffer = ReplayBuffer(100000, device=device)
cur_frame = 0
num_iters = 400
updates = 0
reward_record = np.zeros((1000,))
history_record = np.zeros((10,400,num_ue))
# Start training. Play game once and then train with a batch.
last_100_ep_rewards = []
for episode in range(num_episodes+1):
#   state = np.concatenate((np.reshape(H_r[0,:,:],(1,-1)),np.reshape(H_i[0,:,:],(1,-1))),axis = 1)
  state = np.array(np.reshape(se_max_ur[0,:],(1,num_ue)))
  state = state.astype(np.float32)
  ep_reward, done = 0, False
  episode_steps = 0
  ue_history = 0.01*np.ones((num_ue,))
  while not done:
    state_in = torch.from_numpy(np.expand_dims(state, axis=0)).to(device)
    action = select_epsilon_greedy_action(state_in, epsilon)
    ue_select,idx = sel_ue(action)

    mod_select = np.ones((idx,)) *16
    ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(H[episode_steps,:,ue_select],(num_bs,-1)),idx,mod_select)
    # ur_se_total = ur_se_total/normal_ur[episode_steps]
    for i in range(0,idx):
        ue_history[ue_select[i]] += ur_se[i]
    
    if (episode >= 990):
                history_record[episode-990,episode_steps,:] = ue_history

    jfi = np.square((np.sum(ue_history))) / (num_ue * np.sum(np.square(ue_history)))
    if episode_steps == num_iters -2:
        print('jfi is', episode_steps, jfi)

    # next_state = np.concatenate((np.reshape(H_r[(episode_steps+1),:,:],(1,-1)),np.reshape(H_i[(episode_steps+1),:,:],(1,-1))),axis = 1)
    next_state = np.array(np.reshape(se_max_ur[(episode_steps+1),:],(1,num_ue)))
    next_state = next_state.astype(np.float32)
    reward  = (a*ur_se_total)*reward_scale
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
    # Copy main_nn weights to target_nn.
    if cur_frame % 2000 == 0:
      target_nn.load_state_dict(main_nn.state_dict())
    
    # Train neural network.
    if len(buffer) > batch_size:
      updates += 1 
      states, actions, rewards, next_states, dones = buffer.sample(batch_size)
      loss = train_step(states, actions, rewards, next_states, dones)
      writer.add_scalar('loss', loss, updates)

  if episode < 950:
    epsilon -= 0.001
  
  writer.add_scalar('reward', ep_reward, episode)
  reward_record[episode] = ep_reward

  if len(last_100_ep_rewards) == 100:
    last_100_ep_rewards = last_100_ep_rewards[1:]
  last_100_ep_rewards.append(ep_reward)

  if episode % 50 == 0:
    print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}.'
          f' Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.2f}')
    torch.save(main_nn, 'checkpoint_dqn.pth')

with h5py.File("./results/4_2_4_DQN_mobile.hdf5", "w") as data_file:
    data_file.create_dataset("reward", data=reward_record)
    data_file.create_dataset("history", data=history_record)
    # data_file.create_dataset("max_history", data=max_history_record)
print('Training is finished\n')

# env.close()

# """# Display Result of Trained DQN Agent on Cartpole Environment"""

# def show_video():
#   """Enables video recording of gym environment and shows it."""
#   mp4list = glob.glob('video/*.mp4')
#   if len(mp4list) > 0:
#     mp4 = mp4list[0]
#     video = io.open(mp4, 'r+b').read()
#     encoded = base64.b64encode(video)
#     ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
#                 loop controls style="height: 400px;">
#                 <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#              </video>'''.format(encoded.decode('ascii'))))
#   else: 
#     print("Video not found")

# def wrap_env(env):
#   env = Monitor(env, './video', force=True)
#   return env

# env = wrap_env(gym.make('CartPole-v0'))
# state = env.reset()
# done = False
# ep_rew = 0
# while not done:
#   env.render()
#   state = state.astype(np.float32)
#   state = torch.from_numpy(np.expand_dims(state, axis=0)).to(device)
#   action = select_epsilon_greedy_action(state, epsilon=0.01)
#   state, reward, done, info = env.step(action)
#   ep_rew += reward
# print('Return on this episode: {}'.format(ep_rew))
# env.close()
# show_video()
