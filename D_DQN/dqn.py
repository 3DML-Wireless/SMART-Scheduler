import random
import datetime
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent
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

agent = Agent(state_size=40, action_size=10, seed=0)

#Tensorboard
writer = SummaryWriter('runs/{}_DQN'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

sel_ue = 2
num_ue = 4
num_bs = 4
# randoms = 1.
# epsilon = 20000

user_set = [0,1,2,3]
a = 1
b = 1
c = 0
reward_scale = 20
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




# watch an untrained agent
# state = env.reset()
# for j in range(200):
#     action = agent.act(state)
#     env.render()
#     state, reward, done, _ = env.step(action)
#     if done:
#         break 


def dqn(n_episodes=800, max_t=400, randoms = 1., epsilon = 80000):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = randoms
    # updates = 0       
    for i_episode in range(0, n_episodes):
        # group_idx = usr_group(np.squeeze(H[0,:,:]))
        score = 0
        done = False
        episode_steps = 0
        ue_ep_history = np.zeros((400,num_ue))
        ue_history = 0.01*np.ones((num_ue,))
        
        # state = np.concatenate((np.reshape(se_max_ur[0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(group_idx,(1,-1))),axis = 1)
        state = np.concatenate((np.reshape(se_max_ur[0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_r[0,:,:],(1,-1)),np.reshape(H_i[0,:,:],(1,-1))),axis = 1)
        while not done:
            # print('Training processing: ',i_episode,' episode ',episode_steps,' episode_steps')
            start_time = time.time()

            action = agent.act(state, eps)
            # print("Current action:", action)
            ue_select,idx = sel_ue(action)

            mod_select = np.ones((idx,)) *16
            ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(H[episode_steps,:,ue_select],(num_bs,-1)),idx,mod_select)
            ur_se_total = ur_se_total/normal_ur[episode_steps]
            eps -= 1/epsilon
            if eps < 0:
                eps = 0
            print("eps:",eps)
            # print('se_total:',ur_se_total)
            # print('idx is',idx)
            # print('ue_selection is:', ue_select)
            
            for i in range(0,idx):
                ue_history[ue_select[i]] += ur_se[i]
            ue_ep_history[episode_steps,:] = ue_history
            if (i_episode >= 790):
                history_record[i_episode-790,episode_steps,:] = ue_history

            jfi = np.square((np.sum(ue_history))) / (num_ue * np.sum(np.square(ue_history)))
            print('jfi is', jfi)
            # group_idx_next = usr_group(np.squeeze(H[(episode_steps+1),:,:]))
            next_state = np.concatenate((np.reshape(se_max_ur[(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_r[(episode_steps+1),:,:],(1,-1)),np.reshape(H_i[(episode_steps+1),:,:],(1,-1))),axis = 1)
            
            reward  = (a*ur_se_total + b*jfi)*reward_scale
            done = False
            # print('reward is:', reward)
            if max_t and episode_steps >= max_t-2:
                done = True
            # print('reward is:', reward)


            # next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            # writer.add_scalar('loss', loss, updates)
            
            state = next_state
            score += reward
            

            episode_steps += 1
            # updates += 1
            end_time = time.time()
            # print('Training time is:', (end_time-start_time), '\n')

        
        reward_record[i_episode] = score
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        # eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)),'\n')
        if (i_episode > 500) and (i_episode % 10 == 0):
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)),'\n')
        # if np.mean(scores_window)>=200.0:
        #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        # break
        writer.add_scalar('reward', score, i_episode)
    return scores

scores = dqn()
with h5py.File("./results/4_2_4_DQN_mobile.hdf5", "w") as data_file:
    data_file.create_dataset("reward", data=reward_record)
    data_file.create_dataset("history", data=history_record)
    # data_file.create_dataset("max_history", data=max_history_record)

print('Training is finished\n')


# plot the scores
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()


### Test
# agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

# for i in range(3):
#     state = env.reset()
#     for j in range(200):
#         action = agent.act(state)
#         env.render()
#         state, reward, done, _ = env.step(action)
#         if done:
#             break 