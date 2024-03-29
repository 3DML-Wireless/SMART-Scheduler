import argparse
import datetime
import time
import numpy as np
from itertools import combinations
import torch
import h5py
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
# from mimo_data_process import *
from mimo_sim_ul import *
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="QuaDriGa_SAC_KNN",
                    help='QuaDriGa_SAC_KNN')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha_lr', type=float, default=0.0001, metavar='G',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: True)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--max_episode_steps', type=int, default=400, metavar='N',
                    help='maximum number of steps (TTI) (default: 1000)')
parser.add_argument('--max_episode', type=int, default=1000, metavar='N',
                    help='maximum number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--save_per_epochs', type=int, default=15, metavar='N',
                    help='save_per_epochs')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', default = 0,
                    help='run on CUDA (default: False)')
parser.add_argument('--gpu_nums', type=int, default=2, help='#GPUs to use (default: 1)')

args = parser.parse_args()

# Environment

torch.manual_seed(args.seed)
np.random.seed(args.seed)



### Import data from hdf5 file #############################

# Import se_max and H

# data_shuffled_file  = h5py.File('../channel_sequences.hdf5', 'r')
# se_max_ur = data_shuffled_file.get('se_max')
# se_max_ur = np.array(se_max_ur)
# # print('se data type:',se_max_ur.dtype)
# H_r = np.array(data_shuffled_file.get('H_r'))
# H_i = np.array(data_shuffled_file.get('H_i'))
# H_i = np.transpose(H_i,(2,1,0))
# H_r = np.transpose(H_r,(2,1,0))
# # print("H_r shape is:", H_r.shape)
# # print("H_i shape is:", H_i.shape)
# # print('H_r H_i type:',H_r.dtype, H_i.dtype)
# H = np.array(H_r + 1j*H_i)
# print("H shape is:", H.shape)
# print("se_max shape is:", se_max_ur.shape)

H_file = h5py.File('./16_4_16_2RBmob_normal.hdf5','r')
H_r = np.array(H_file.get('H_r_1'))
H_i = np.array(H_file.get('H_i_1'))
# H_i = np.transpose(H_i,(2,1,0))
# H_r = np.transpose(H_r,(2,1,0))
# print("H_r_1 shape is:", H_r_1.shape)
# print("H_i_1 shape is:", H_i_1.shape)
H = np.array(H_r + 1j*H_i)
print("H_1 shape is:", H.shape)

se_max_ur = np.array(H_file.get('se_max_1'))
print("se_max_ur shape is:", se_max_ur.shape)

normal_ur = np.array(H_file.get('normal_1'))
print("normal_ur shape is:", normal_ur.shape)

#################################################################
#### Single RB
# H_file = h5py.File('./16_4_16_mobile_all.hdf5','r')
# H_r = np.array(H_file.get('H_r'))
# H_i = np.array(H_file.get('H_i'))
# # H_i = np.transpose(H_i,(2,1,0))
# # H_r = np.transpose(H_r,(2,1,0))
# print("H_r shape is:", H_r.shape)
# print("H_i shape is:", H_i.shape)
# H = np.array(H_r + 1j*H_i)
# print("H shape is:", H.shape)

# se_max_ur = H_file.get('se_max')
# se_max_ur = np.array(se_max_ur)
# print("se_max_ur shape is:", se_max_ur.shape)


# N_file = h5py.File('./16_4_16_mobile_normal.hdf5','r')
# normal_ur = np.array(N_file.get('normal'))
# print("normal_ur shape is:", normal_ur.shape)

# pre_file = h5py.File('./plot_data/pre_processing_16_4_16_mobile.hdf5','r')
# pre_data = np.array(pre_file.get('pre'))
# print("pre_data shape is:", pre_data.shape)
#############################################################
sel_ue = 4
num_ue = 16
num_bs = 16

max_actions = 2516
num_states = 544
num_actions = 1
random = 1
epsilon = 20000

# Agent
agent = SAC(num_states, num_actions, max_actions, args, args.lr,args.alpha_lr)

#Tensorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

### User space
user_set = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

### Reward Function Parameters
a = 5
b = 1
c = 0
reward_scale = 1
### Record vector defination
reward_record = np.zeros((1000,))
history_record = np.zeros((10,400,num_ue))
max_history_record = np.zeros((400,num_ue))
ue_history = 0.01*np.ones((num_ue,)) ## Tp history for each UE [4,]

def sel_ue(action):
    sum_before = 0
    for i in range (1,5):
        sum_before += len(list(combinations(user_set, i)))
        if ((action+1)>sum_before):
            continue
        else:
            idx = i
            sum_before -= len(list(combinations(user_set, i)))
            ue_select = list(combinations(user_set, i))[action-sum_before]
            break
    return ue_select,idx


# start_time = time.time()
for i_episode in range (args.max_episode): 
    episode_reward = 0
    episode_steps = 0
    done = False
    ue_ep_history = np.zeros((400,num_ue))
    ue_history = 0.01*np.ones((num_ue,))
    state = np.concatenate((np.reshape(se_max_ur[0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_r[0,:,:],(1,-1)),np.reshape(H_i[0,:,:],(1,-1))),axis = 1)
    print('state shape is:',state.shape)
    while not done:
        print('Training processing: ',i_episode,' episode ',episode_steps,' episode_steps')
        start_time = time.time()
        if random > np.random.rand(1):
            # if step <= warmup or (episode < 100):
            action, final_action = agent.random_action()
            print('Random action',final_action)
        else:
            action, final_action = agent.select_action(state)
            print('Actor action',final_action)
        # print('final action is: ', final_action)
        inf_time = time.time()
        print("Inf time is:",inf_time-start_time)
        random -= 1/epsilon
        
        # if args.start_steps > total_numsteps:
        #     action, final_action = agent.random_action()
        #     # action = env.action_space.sample()  # # Modify !!!!!!!!!!!!!!!!!
        # else:
        #     action, final_action = agent.select_action(state)  # Sample action from policy
        # print('final action is: ', final_action)

        ue_select,idx = sel_ue(final_action[0])
        # ur_se_total = pre_data[episode_steps,final_action[0],0]
        # ur_min_snr = pre_data[episode_steps,final_action[0],1]
        # ur_se = pre_data[episode_steps,final_action[0],2:2+idx]
        mod_select = np.ones((idx,)) *16 # 16-QAM
        ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(H[episode_steps,:,ue_select],(num_bs,-1)),idx,mod_select)
        ur_se_total = ur_se_total/normal_ur[episode_steps]
        print('se_total, min_snr are',ur_se_total, ur_min_snr)
        print('idx is',idx)
        print('ue_selection is:', ue_select)
        for i in range(0,idx):
            ue_history[ue_select[i]] += ur_se[i]
            # print('ur_se is',ur_se[i])
        
        ue_ep_history[episode_steps,:] = ue_history

        if (i_episode >= 990):
            history_record[i_episode-990,episode_steps,:] = ue_history
       
        jfi = np.square((np.sum(ue_history))) / (num_ue * np.sum(np.square(ue_history)))
        print('jfi is', jfi)
        next_state = np.concatenate((np.reshape(se_max_ur[(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_r[(episode_steps+1),:,:],(1,-1)),np.reshape(H_i[(episode_steps+1),:,:],(1,-1))),axis = 1)
        reward  = (a*ur_se_total + b*jfi + c*ur_min_snr)*reward_scale
        done = False
        print('reward is:', reward)
        if args.max_episode_steps and episode_steps >= args.max_episode_steps-2:
            done = True
            
        cal_time = time.time()
        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1
        
        # next_state, reward, done, _ = env.step(final_action) # Modify !!!!!!!!!!!!!!!!!
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
        print('episode reward is:', episode_reward,'\n')
        
        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps >= args.max_episode_steps -1 else float(not done)
        print('mask is:',mask)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state
        end_time = time.time()
        print('Training time is:', (end_time-cal_time+inf_time-start_time))

    # if total_numsteps > args.num_steps:
    #     break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    reward_record[i_episode] = episode_reward
    if episode_reward == np.max(reward_record):
        max_history_record = ue_ep_history

    if args.start_steps < total_numsteps and i_episode > 0 and i_episode % args.save_per_epochs == 0:
        agent.save_checkpoint('16_4_16_vanilla_2RBmob_sep')


    # if i_episode % 10 == 0 and args.eval is True:
    #     avg_reward = 0.
    #     episodes = 10
    #     for _  in range(episodes):
    #         state = env.reset()
    #         episode_reward = 0
    #         done = False
    #         while not done:
    #             action = agent.select_action(state, evaluate=True)

    #             next_state, reward, done, _ = env.step(action)
    #             episode_reward += reward


    #             state = next_state
    #         avg_reward += episode_reward
    #     avg_reward /= episodes


    #     writer.add_scalar('avg_reward/test', avg_reward, i_episode)

    #     print("----------------------------------------")
    #     print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
    #     print("----------------------------------------")


with h5py.File("./data/16_4_16_vanilla_1_1_mobile.hdf5", "w") as data_file:
    data_file.create_dataset("reward", data=reward_record)
    data_file.create_dataset("history", data=history_record)
    data_file.create_dataset("max_history", data=max_history_record)

print('Training is finished\n')
# end_time = time.time()
# print('Training time is:', (end_time-start_time))