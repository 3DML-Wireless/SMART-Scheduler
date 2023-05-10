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
parser.add_argument('--lr_1', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha_lr_1', type=float, default=0.0001, metavar='G',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--lr_2', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha_lr_2', type=float, default=0.0001, metavar='G',
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
parser.add_argument('--max_episode', type=int, default=800, metavar='N',
                    help='maximum number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
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
parser.add_argument('--cuda', default = 1,
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



H_file = h5py.File('./8_8_50RB_static_normal.hdf5','r')
H_rc = np.array(H_file.get('H_rc'))
H_ic = np.array(H_file.get('H_ic'))

H_c = np.array(H_rc + 1j*H_ic)
print("H_c shape is:", H_c.shape)

se_max_ur = np.array(H_file.get('se_max'))
print("se_max_ur shape is:", se_max_ur.shape)

normal_ur = np.array(H_file.get('normal'))
print("normal_ur shape is:", normal_ur.shape)
#############################################################
sel_ue = 4
num_ue = 8
num_bs = 8

max_actions = 162
num_states = 144
num_actions = 1
random = 0
epsilon = 20000

# Agent
agent_1 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_2 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_3 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_4 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_5 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_6 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_7 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_8 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_9 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_10 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)

agent_11 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_12 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_13 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_14 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_15 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_16 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_17 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_18 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_19 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_20 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)

agent_21 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_22 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_23 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_24 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_25 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_26 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_27 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_28 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_29 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_30 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)

agent_31 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_32 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_33 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_34 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_35 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_36 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_37 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_38 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_39 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_40 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)

agent_41 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_42 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_43 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_44 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_45 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_46 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_47 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_48 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_49 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)
agent_50 = SAC(num_states, num_actions, max_actions, args, 0.00005, 0.00005)

#Tensorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory_1 = ReplayMemory(args.replay_size, args.seed)
memory_2 = ReplayMemory(args.replay_size, args.seed)
memory_3 = ReplayMemory(args.replay_size, args.seed)
memory_4 = ReplayMemory(args.replay_size, args.seed)
memory_5 = ReplayMemory(args.replay_size, args.seed)
memory_6 = ReplayMemory(args.replay_size, args.seed)
memory_7 = ReplayMemory(args.replay_size, args.seed)
memory_8 = ReplayMemory(args.replay_size, args.seed)
memory_9 = ReplayMemory(args.replay_size, args.seed)
memory_10 = ReplayMemory(args.replay_size, args.seed)

memory_11 = ReplayMemory(args.replay_size, args.seed)
memory_12 = ReplayMemory(args.replay_size, args.seed)
memory_13 = ReplayMemory(args.replay_size, args.seed)
memory_14 = ReplayMemory(args.replay_size, args.seed)
memory_15 = ReplayMemory(args.replay_size, args.seed)
memory_16 = ReplayMemory(args.replay_size, args.seed)
memory_17 = ReplayMemory(args.replay_size, args.seed)
memory_18 = ReplayMemory(args.replay_size, args.seed)
memory_19 = ReplayMemory(args.replay_size, args.seed)
memory_20 = ReplayMemory(args.replay_size, args.seed)

memory_21 = ReplayMemory(args.replay_size, args.seed)
memory_22 = ReplayMemory(args.replay_size, args.seed)
memory_23 = ReplayMemory(args.replay_size, args.seed)
memory_24 = ReplayMemory(args.replay_size, args.seed)
memory_25 = ReplayMemory(args.replay_size, args.seed)
memory_26 = ReplayMemory(args.replay_size, args.seed)
memory_27 = ReplayMemory(args.replay_size, args.seed)
memory_28 = ReplayMemory(args.replay_size, args.seed)
memory_29 = ReplayMemory(args.replay_size, args.seed)
memory_30 = ReplayMemory(args.replay_size, args.seed)

memory_31 = ReplayMemory(args.replay_size, args.seed)
memory_32 = ReplayMemory(args.replay_size, args.seed)
memory_33 = ReplayMemory(args.replay_size, args.seed)
memory_34 = ReplayMemory(args.replay_size, args.seed)
memory_35 = ReplayMemory(args.replay_size, args.seed)
memory_36 = ReplayMemory(args.replay_size, args.seed)
memory_37 = ReplayMemory(args.replay_size, args.seed)
memory_38 = ReplayMemory(args.replay_size, args.seed)
memory_39 = ReplayMemory(args.replay_size, args.seed)
memory_40 = ReplayMemory(args.replay_size, args.seed)

memory_41 = ReplayMemory(args.replay_size, args.seed)
memory_42 = ReplayMemory(args.replay_size, args.seed)
memory_43 = ReplayMemory(args.replay_size, args.seed)
memory_44 = ReplayMemory(args.replay_size, args.seed)
memory_45 = ReplayMemory(args.replay_size, args.seed)
memory_46 = ReplayMemory(args.replay_size, args.seed)
memory_47 = ReplayMemory(args.replay_size, args.seed)
memory_48 = ReplayMemory(args.replay_size, args.seed)
memory_49 = ReplayMemory(args.replay_size, args.seed)
memory_50 = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

### User space
user_set = [0,1,2,3,4,5,6,7]

### Reward Function Parameters
a = 2
b = 1
c = 0
reward_scale = 1
### Record vector defination
# reward_record_1 = np.zeros((1000,))
history_record = np.zeros((400,num_ue))
# max_history_record = np.zeros((400,num_ue))

ue_history = 0.01 * np.ones((num_ue,))
# for i in range (0,50):
#     ue_history[i] = 0.01*np.ones((num_ue,)) ## Tp history for each UE [4,]

# reward_record_2 = np.zeros((1000,))
# history_record_2 = np.zeros((10,400,num_ue))
# max_history_record_2 = np.zeros((400,num_ue))
# ue_history = 0.01*np.ones((num_ue,)) ## Tp history for each UE [4,]

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
for i_episode in range (0,1): 
    episode_reward = np.zeros((50,))
    episode_steps = 0
    done = False
    # ue_ep_history = np.zeros((400,num_ue))
    # ue_history = 0.01*np.ones((num_ue,))
    # ckpt_path_1 = "checkpoints/sac_checkpoint_16_4_16_vanilla_2RBmob_sep_"
    ckpt_path_1 = "checkpoints/sac_checkpoint_8_8_50RB_static_ind_new_"
    

    agent_1.load_checkpoint(ckpt_path_1)
    agent_2.load_checkpoint(ckpt_path_1)
    agent_3.load_checkpoint(ckpt_path_1)
    agent_4.load_checkpoint(ckpt_path_1)
    agent_5.load_checkpoint(ckpt_path_1)
    agent_6.load_checkpoint(ckpt_path_1)
    agent_7.load_checkpoint(ckpt_path_1)
    agent_8.load_checkpoint(ckpt_path_1)
    agent_9.load_checkpoint(ckpt_path_1)
    agent_10.load_checkpoint(ckpt_path_1)

    agent_11.load_checkpoint(ckpt_path_1)
    agent_12.load_checkpoint(ckpt_path_1)
    agent_13.load_checkpoint(ckpt_path_1)
    agent_14.load_checkpoint(ckpt_path_1)
    agent_15.load_checkpoint(ckpt_path_1)
    agent_16.load_checkpoint(ckpt_path_1)
    agent_17.load_checkpoint(ckpt_path_1)
    agent_18.load_checkpoint(ckpt_path_1)
    agent_19.load_checkpoint(ckpt_path_1)
    agent_20.load_checkpoint(ckpt_path_1)

    agent_21.load_checkpoint(ckpt_path_1)
    agent_22.load_checkpoint(ckpt_path_1)
    agent_23.load_checkpoint(ckpt_path_1)
    agent_24.load_checkpoint(ckpt_path_1)
    agent_25.load_checkpoint(ckpt_path_1)
    agent_26.load_checkpoint(ckpt_path_1)
    agent_27.load_checkpoint(ckpt_path_1)
    agent_28.load_checkpoint(ckpt_path_1)
    agent_29.load_checkpoint(ckpt_path_1)
    agent_30.load_checkpoint(ckpt_path_1)

    agent_31.load_checkpoint(ckpt_path_1)
    agent_32.load_checkpoint(ckpt_path_1)
    agent_33.load_checkpoint(ckpt_path_1)
    agent_34.load_checkpoint(ckpt_path_1)
    agent_35.load_checkpoint(ckpt_path_1)
    agent_36.load_checkpoint(ckpt_path_1)
    agent_37.load_checkpoint(ckpt_path_1)
    agent_38.load_checkpoint(ckpt_path_1)
    agent_39.load_checkpoint(ckpt_path_1)
    agent_40.load_checkpoint(ckpt_path_1)

    agent_41.load_checkpoint(ckpt_path_1)
    agent_42.load_checkpoint(ckpt_path_1)
    agent_43.load_checkpoint(ckpt_path_1)
    agent_44.load_checkpoint(ckpt_path_1)
    agent_45.load_checkpoint(ckpt_path_1)
    agent_46.load_checkpoint(ckpt_path_1)
    agent_47.load_checkpoint(ckpt_path_1)
    agent_48.load_checkpoint(ckpt_path_1)
    agent_49.load_checkpoint(ckpt_path_1)
    agent_50.load_checkpoint(ckpt_path_1)

    state_1 = np.concatenate((np.reshape(se_max_ur[0,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[0,0,:,:],(1,-1)),np.reshape(H_ic[0,0,:,:],(1,-1))),axis = 1)
    state_2 = np.concatenate((np.reshape(se_max_ur[1,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[1,0,:,:],(1,-1)),np.reshape(H_ic[1,0,:,:],(1,-1))),axis = 1)
    state_3 = np.concatenate((np.reshape(se_max_ur[2,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[2,0,:,:],(1,-1)),np.reshape(H_ic[2,0,:,:],(1,-1))),axis = 1)
    state_4 = np.concatenate((np.reshape(se_max_ur[3,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[3,0,:,:],(1,-1)),np.reshape(H_ic[3,0,:,:],(1,-1))),axis = 1)
    state_5 = np.concatenate((np.reshape(se_max_ur[4,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[4,0,:,:],(1,-1)),np.reshape(H_ic[4,0,:,:],(1,-1))),axis = 1)
    state_6 = np.concatenate((np.reshape(se_max_ur[5,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[5,0,:,:],(1,-1)),np.reshape(H_ic[5,0,:,:],(1,-1))),axis = 1)
    state_7 = np.concatenate((np.reshape(se_max_ur[6,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[6,0,:,:],(1,-1)),np.reshape(H_ic[6,0,:,:],(1,-1))),axis = 1)
    state_8 = np.concatenate((np.reshape(se_max_ur[7,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[7,0,:,:],(1,-1)),np.reshape(H_ic[7,0,:,:],(1,-1))),axis = 1)
    state_9 = np.concatenate((np.reshape(se_max_ur[8,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[8,0,:,:],(1,-1)),np.reshape(H_ic[8,0,:,:],(1,-1))),axis = 1)
    state_10 = np.concatenate((np.reshape(se_max_ur[9,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[9,0,:,:],(1,-1)),np.reshape(H_ic[9,0,:,:],(1,-1))),axis = 1)

    state_11 = np.concatenate((np.reshape(se_max_ur[10,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[10,0,:,:],(1,-1)),np.reshape(H_ic[10,0,:,:],(1,-1))),axis = 1)
    state_12 = np.concatenate((np.reshape(se_max_ur[11,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[11,0,:,:],(1,-1)),np.reshape(H_ic[11,0,:,:],(1,-1))),axis = 1)
    state_13 = np.concatenate((np.reshape(se_max_ur[12,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[12,0,:,:],(1,-1)),np.reshape(H_ic[12,0,:,:],(1,-1))),axis = 1)
    state_14 = np.concatenate((np.reshape(se_max_ur[13,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[13,0,:,:],(1,-1)),np.reshape(H_ic[13,0,:,:],(1,-1))),axis = 1)
    state_15 = np.concatenate((np.reshape(se_max_ur[14,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[14,0,:,:],(1,-1)),np.reshape(H_ic[14,0,:,:],(1,-1))),axis = 1)
    state_16 = np.concatenate((np.reshape(se_max_ur[15,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[15,0,:,:],(1,-1)),np.reshape(H_ic[15,0,:,:],(1,-1))),axis = 1)
    state_17 = np.concatenate((np.reshape(se_max_ur[16,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[16,0,:,:],(1,-1)),np.reshape(H_ic[16,0,:,:],(1,-1))),axis = 1)
    state_18 = np.concatenate((np.reshape(se_max_ur[17,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[17,0,:,:],(1,-1)),np.reshape(H_ic[17,0,:,:],(1,-1))),axis = 1)
    state_19 = np.concatenate((np.reshape(se_max_ur[18,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[18,0,:,:],(1,-1)),np.reshape(H_ic[18,0,:,:],(1,-1))),axis = 1)
    state_20 = np.concatenate((np.reshape(se_max_ur[19,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[19,0,:,:],(1,-1)),np.reshape(H_ic[19,0,:,:],(1,-1))),axis = 1)

    state_21 = np.concatenate((np.reshape(se_max_ur[20,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[20,0,:,:],(1,-1)),np.reshape(H_ic[20,0,:,:],(1,-1))),axis = 1)
    state_22 = np.concatenate((np.reshape(se_max_ur[21,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[21,0,:,:],(1,-1)),np.reshape(H_ic[21,0,:,:],(1,-1))),axis = 1)
    state_23 = np.concatenate((np.reshape(se_max_ur[22,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[22,0,:,:],(1,-1)),np.reshape(H_ic[22,0,:,:],(1,-1))),axis = 1)
    state_24 = np.concatenate((np.reshape(se_max_ur[23,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[23,0,:,:],(1,-1)),np.reshape(H_ic[23,0,:,:],(1,-1))),axis = 1)
    state_25 = np.concatenate((np.reshape(se_max_ur[24,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[24,0,:,:],(1,-1)),np.reshape(H_ic[24,0,:,:],(1,-1))),axis = 1)
    state_26 = np.concatenate((np.reshape(se_max_ur[25,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[25,0,:,:],(1,-1)),np.reshape(H_ic[25,0,:,:],(1,-1))),axis = 1)
    state_27 = np.concatenate((np.reshape(se_max_ur[26,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[26,0,:,:],(1,-1)),np.reshape(H_ic[26,0,:,:],(1,-1))),axis = 1)
    state_28 = np.concatenate((np.reshape(se_max_ur[27,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[27,0,:,:],(1,-1)),np.reshape(H_ic[27,0,:,:],(1,-1))),axis = 1)
    state_29 = np.concatenate((np.reshape(se_max_ur[28,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[28,0,:,:],(1,-1)),np.reshape(H_ic[28,0,:,:],(1,-1))),axis = 1)
    state_30 = np.concatenate((np.reshape(se_max_ur[29,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[29,0,:,:],(1,-1)),np.reshape(H_ic[29,0,:,:],(1,-1))),axis = 1)

    state_31 = np.concatenate((np.reshape(se_max_ur[30,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[30,0,:,:],(1,-1)),np.reshape(H_ic[30,0,:,:],(1,-1))),axis = 1)
    state_32 = np.concatenate((np.reshape(se_max_ur[31,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[31,0,:,:],(1,-1)),np.reshape(H_ic[31,0,:,:],(1,-1))),axis = 1)
    state_33 = np.concatenate((np.reshape(se_max_ur[32,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[32,0,:,:],(1,-1)),np.reshape(H_ic[32,0,:,:],(1,-1))),axis = 1)
    state_34 = np.concatenate((np.reshape(se_max_ur[33,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[33,0,:,:],(1,-1)),np.reshape(H_ic[33,0,:,:],(1,-1))),axis = 1)
    state_35 = np.concatenate((np.reshape(se_max_ur[34,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[34,0,:,:],(1,-1)),np.reshape(H_ic[34,0,:,:],(1,-1))),axis = 1)
    state_36 = np.concatenate((np.reshape(se_max_ur[35,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[35,0,:,:],(1,-1)),np.reshape(H_ic[35,0,:,:],(1,-1))),axis = 1)
    state_37 = np.concatenate((np.reshape(se_max_ur[36,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[36,0,:,:],(1,-1)),np.reshape(H_ic[36,0,:,:],(1,-1))),axis = 1)
    state_38 = np.concatenate((np.reshape(se_max_ur[37,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[37,0,:,:],(1,-1)),np.reshape(H_ic[37,0,:,:],(1,-1))),axis = 1)
    state_39 = np.concatenate((np.reshape(se_max_ur[38,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[38,0,:,:],(1,-1)),np.reshape(H_ic[38,0,:,:],(1,-1))),axis = 1)
    state_40 = np.concatenate((np.reshape(se_max_ur[39,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[39,0,:,:],(1,-1)),np.reshape(H_ic[39,0,:,:],(1,-1))),axis = 1)

    state_41 = np.concatenate((np.reshape(se_max_ur[40,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[40,0,:,:],(1,-1)),np.reshape(H_ic[40,0,:,:],(1,-1))),axis = 1)
    state_42 = np.concatenate((np.reshape(se_max_ur[41,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[41,0,:,:],(1,-1)),np.reshape(H_ic[41,0,:,:],(1,-1))),axis = 1)
    state_43 = np.concatenate((np.reshape(se_max_ur[42,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[42,0,:,:],(1,-1)),np.reshape(H_ic[42,0,:,:],(1,-1))),axis = 1)
    state_44 = np.concatenate((np.reshape(se_max_ur[43,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[43,0,:,:],(1,-1)),np.reshape(H_ic[43,0,:,:],(1,-1))),axis = 1)
    state_45 = np.concatenate((np.reshape(se_max_ur[44,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[44,0,:,:],(1,-1)),np.reshape(H_ic[44,0,:,:],(1,-1))),axis = 1)
    state_46 = np.concatenate((np.reshape(se_max_ur[45,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[45,0,:,:],(1,-1)),np.reshape(H_ic[45,0,:,:],(1,-1))),axis = 1)
    state_47 = np.concatenate((np.reshape(se_max_ur[46,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[46,0,:,:],(1,-1)),np.reshape(H_ic[46,0,:,:],(1,-1))),axis = 1)
    state_48 = np.concatenate((np.reshape(se_max_ur[47,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[47,0,:,:],(1,-1)),np.reshape(H_ic[47,0,:,:],(1,-1))),axis = 1)
    state_49 = np.concatenate((np.reshape(se_max_ur[48,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[48,0,:,:],(1,-1)),np.reshape(H_ic[48,0,:,:],(1,-1))),axis = 1)
    state_50 = np.concatenate((np.reshape(se_max_ur[49,0,:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[49,0,:,:],(1,-1)),np.reshape(H_ic[49,0,:,:],(1,-1))),axis = 1)

    #print('state shape is:',state_1.shape, state_2.shape)
    while not done:
        print('Training processing: ',i_episode,' episode ',episode_steps,' episode_steps')
        start_time = time.time()
        if random > np.random.rand(1):
            # if step <= warmup or (episode < 100):
            action_1, final_action_1 = agent_1.random_action()
            action_2, final_action_2 = agent_2.random_action()
            action_3, final_action_3 = agent_3.random_action()
            action_4, final_action_4 = agent_4.random_action()
            action_5, final_action_5 = agent_5.random_action()
            action_6, final_action_6 = agent_6.random_action()
            action_7, final_action_7 = agent_7.random_action()
            action_8, final_action_8 = agent_8.random_action()
            action_9, final_action_9 = agent_9.random_action()
            action_10, final_action_10 = agent_10.random_action()

            action_11, final_action_11 = agent_11.random_action()
            action_12, final_action_12 = agent_12.random_action()
            action_13, final_action_13 = agent_13.random_action()
            action_14, final_action_14 = agent_14.random_action()
            action_15, final_action_15 = agent_15.random_action()
            action_16, final_action_16 = agent_16.random_action()
            action_17, final_action_17 = agent_17.random_action()
            action_18, final_action_18 = agent_18.random_action()
            action_19, final_action_19 = agent_19.random_action()
            action_20, final_action_20 = agent_20.random_action()

            action_21, final_action_21 = agent_21.random_action()
            action_22, final_action_22 = agent_22.random_action()
            action_23, final_action_23 = agent_23.random_action()
            action_24, final_action_24 = agent_24.random_action()
            action_25, final_action_25 = agent_25.random_action()
            action_26, final_action_26 = agent_26.random_action()
            action_27, final_action_27 = agent_27.random_action()
            action_28, final_action_28 = agent_28.random_action()
            action_29, final_action_29 = agent_29.random_action()
            action_30, final_action_30 = agent_30.random_action()

            action_31, final_action_31 = agent_31.random_action()
            action_32, final_action_32 = agent_32.random_action()
            action_33, final_action_33 = agent_33.random_action()
            action_34, final_action_34 = agent_34.random_action()
            action_35, final_action_35 = agent_35.random_action()
            action_36, final_action_36 = agent_36.random_action()
            action_37, final_action_37 = agent_37.random_action()
            action_38, final_action_38 = agent_38.random_action()
            action_39, final_action_39 = agent_39.random_action()
            action_40, final_action_40 = agent_40.random_action()

            action_41, final_action_41 = agent_41.random_action()
            action_42, final_action_42 = agent_42.random_action()
            action_43, final_action_43 = agent_43.random_action()
            action_44, final_action_44 = agent_44.random_action()
            action_45, final_action_45 = agent_45.random_action()
            action_46, final_action_46 = agent_46.random_action()
            action_47, final_action_47 = agent_47.random_action()
            action_48, final_action_48 = agent_48.random_action()
            action_49, final_action_49 = agent_49.random_action()
            action_50, final_action_50 = agent_50.random_action()


            print('Random action \n')
        else:
            # action_1, final_action_1 = agent_1.select_action(state_1)
            # action_2, final_action_2 = agent_2.select_action(state_2)

            action_1, final_action_1 = agent_1.select_action(state_1)
            action_2, final_action_2 = agent_2.select_action(state_2)
            action_3, final_action_3 = agent_3.select_action(state_3)
            action_4, final_action_4 = agent_4.select_action(state_4)
            action_5, final_action_5 = agent_5.select_action(state_5)
            action_6, final_action_6 = agent_6.select_action(state_6)
            action_7, final_action_7 = agent_7.select_action(state_7)
            action_8, final_action_8 = agent_8.select_action(state_8)
            action_9, final_action_9 = agent_9.select_action(state_9)
            action_10, final_action_10 = agent_10.select_action(state_10)

            action_11, final_action_11 = agent_11.select_action(state_11)
            action_12, final_action_12 = agent_12.select_action(state_12)
            action_13, final_action_13 = agent_13.select_action(state_13)
            action_14, final_action_14 = agent_14.select_action(state_14)
            action_15, final_action_15 = agent_15.select_action(state_15)
            action_16, final_action_16 = agent_16.select_action(state_16)
            action_17, final_action_17 = agent_17.select_action(state_17)
            action_18, final_action_18 = agent_18.select_action(state_18)
            action_19, final_action_19 = agent_19.select_action(state_19)
            action_20, final_action_20 = agent_20.select_action(state_20)

            action_21, final_action_21 = agent_21.select_action(state_21)
            action_22, final_action_22 = agent_22.select_action(state_22)
            action_23, final_action_23 = agent_23.select_action(state_23)
            action_24, final_action_24 = agent_24.select_action(state_24)
            action_25, final_action_25 = agent_25.select_action(state_25)
            action_26, final_action_26 = agent_26.select_action(state_26)
            action_27, final_action_27 = agent_27.select_action(state_27)
            action_28, final_action_28 = agent_28.select_action(state_28)
            action_29, final_action_29 = agent_29.select_action(state_29)
            action_30, final_action_30 = agent_30.select_action(state_30)

            action_31, final_action_31 = agent_31.select_action(state_31)
            action_32, final_action_32 = agent_32.select_action(state_32)
            action_33, final_action_33 = agent_33.select_action(state_33)
            action_34, final_action_34 = agent_34.select_action(state_34)
            action_35, final_action_35 = agent_35.select_action(state_35)
            action_36, final_action_36 = agent_36.select_action(state_36)
            action_37, final_action_37 = agent_37.select_action(state_37)
            action_38, final_action_38 = agent_38.select_action(state_38)
            action_39, final_action_39 = agent_39.select_action(state_39)
            action_40, final_action_40 = agent_40.select_action(state_40)

            action_41, final_action_41 = agent_41.select_action(state_41)
            action_42, final_action_42 = agent_42.select_action(state_42)
            action_43, final_action_43 = agent_43.select_action(state_43)
            action_44, final_action_44 = agent_44.select_action(state_44)
            action_45, final_action_45 = agent_45.select_action(state_45)
            action_46, final_action_46 = agent_46.select_action(state_46)
            action_47, final_action_47 = agent_47.select_action(state_47)
            action_48, final_action_48 = agent_48.select_action(state_48)
            action_49, final_action_49 = agent_49.select_action(state_49)
            action_50, final_action_50 = agent_50.select_action(state_50)
            print('Actor action \n')
        # print('final action is: ', final_action)
        # pred_time = time.time()
        # print("Prediction time is:",pred_time-start_time)
        # random -= 1/epsilon
        
        # if args.start_steps > total_numsteps:
        #     action, final_action = agent.random_action()
        #     # action = env.action_space.sample()  # # Modify !!!!!!!!!!!!!!!!!
        # else:
        #     action, final_action = agent.select_action(state)  # Sample action from policy
        # print('final action is: ', final_action)

        ue_select_1,idx_1 = sel_ue(final_action_1[0])
        ue_select_2,idx_2 = sel_ue(final_action_2[0])
        ue_select_3,idx_3 = sel_ue(final_action_3[0])
        ue_select_4,idx_4 = sel_ue(final_action_4[0])
        ue_select_5,idx_5 = sel_ue(final_action_5[0])
        ue_select_6,idx_6 = sel_ue(final_action_6[0])
        ue_select_7,idx_7 = sel_ue(final_action_7[0])
        ue_select_8,idx_8 = sel_ue(final_action_8[0])
        ue_select_9,idx_9 = sel_ue(final_action_9[0])
        ue_select_10,idx_10 = sel_ue(final_action_10[0])

        ue_select_11,idx_11 = sel_ue(final_action_1[0])
        ue_select_12,idx_12 = sel_ue(final_action_2[0])
        ue_select_13,idx_13 = sel_ue(final_action_3[0])
        ue_select_14,idx_14 = sel_ue(final_action_4[0])
        ue_select_15,idx_15 = sel_ue(final_action_5[0])
        ue_select_16,idx_16 = sel_ue(final_action_6[0])
        ue_select_17,idx_17 = sel_ue(final_action_7[0])
        ue_select_18,idx_18 = sel_ue(final_action_8[0])
        ue_select_19,idx_19 = sel_ue(final_action_9[0])
        ue_select_20,idx_20 = sel_ue(final_action_10[0])

        ue_select_21,idx_21 = sel_ue(final_action_21[0])
        ue_select_22,idx_22 = sel_ue(final_action_22[0])
        ue_select_23,idx_23 = sel_ue(final_action_23[0])
        ue_select_24,idx_24 = sel_ue(final_action_24[0])
        ue_select_25,idx_25 = sel_ue(final_action_25[0])
        ue_select_26,idx_26 = sel_ue(final_action_26[0])
        ue_select_27,idx_27 = sel_ue(final_action_27[0])
        ue_select_28,idx_28 = sel_ue(final_action_28[0])
        ue_select_29,idx_29 = sel_ue(final_action_29[0])
        ue_select_30,idx_30 = sel_ue(final_action_30[0])

        ue_select_31,idx_31 = sel_ue(final_action_31[0])
        ue_select_32,idx_32 = sel_ue(final_action_32[0])
        ue_select_33,idx_33 = sel_ue(final_action_33[0])
        ue_select_34,idx_34 = sel_ue(final_action_34[0])
        ue_select_35,idx_35 = sel_ue(final_action_35[0])
        ue_select_36,idx_36 = sel_ue(final_action_36[0])
        ue_select_37,idx_37 = sel_ue(final_action_37[0])
        ue_select_38,idx_38 = sel_ue(final_action_38[0])
        ue_select_39,idx_39 = sel_ue(final_action_39[0])
        ue_select_40,idx_40 = sel_ue(final_action_40[0])

        ue_select_41,idx_41 = sel_ue(final_action_41[0])
        ue_select_42,idx_42 = sel_ue(final_action_42[0])
        ue_select_43,idx_43 = sel_ue(final_action_43[0])
        ue_select_44,idx_44 = sel_ue(final_action_44[0])
        ue_select_45,idx_45 = sel_ue(final_action_45[0])
        ue_select_46,idx_46 = sel_ue(final_action_46[0])
        ue_select_47,idx_47 = sel_ue(final_action_47[0])
        ue_select_48,idx_48 = sel_ue(final_action_48[0])
        ue_select_49,idx_49 = sel_ue(final_action_49[0])
        ue_select_50,idx_50 = sel_ue(final_action_50[0]) 

        # ur_se_total = pre_data[episode_steps,final_action[0],0]
        # ur_min_snr = pre_data[episode_steps,final_action[0],1]
        # ur_se = pre_data[episode_steps,final_action[0],2:2+idx]

        mod_select_1 = np.ones((idx_1,)) *16 # 16-QAM
        ur_se_total_1, ur_min_snr_1, ur_se_1 = data_process(np.reshape(H_c[0,episode_steps,:,ue_select_1],(num_bs,-1)),idx_1,mod_select_1)
        ur_se_total_1 = ur_se_total_1/normal_ur[0,episode_steps]

        mod_select_2 = np.ones((idx_2,)) *16
        ur_se_total_2, ur_min_snr_2, ur_se_2 = data_process(np.reshape(H_c[1,episode_steps,:,ue_select_2],(num_bs,-1)),idx_2,mod_select_2)
        ur_se_total_2 = ur_se_total_2/normal_ur[1,episode_steps]

        mod_select_3 = np.ones((idx_3,)) *16 # 36-QAM
        ur_se_total_3, ur_min_snr_3, ur_se_3 = data_process(np.reshape(H_c[2,episode_steps,:,ue_select_3],(num_bs,-1)),idx_3,mod_select_3)
        ur_se_total_3 = ur_se_total_3/normal_ur[2,episode_steps]

        mod_select_4 = np.ones((idx_4,)) *16
        ur_se_total_4, ur_min_snr_4, ur_se_4 = data_process(np.reshape(H_c[3,episode_steps,:,ue_select_4],(num_bs,-1)),idx_4,mod_select_4)
        ur_se_total_4 = ur_se_total_4/normal_ur[3,episode_steps]

        mod_select_5 = np.ones((idx_5,)) *16 # 16-QAM
        ur_se_total_5, ur_min_snr_5, ur_se_5 = data_process(np.reshape(H_c[4,episode_steps,:,ue_select_5],(num_bs,-1)),idx_5,mod_select_5)
        ur_se_total_5 = ur_se_total_5/normal_ur[4,episode_steps]

        mod_select_6 = np.ones((idx_6,)) *16
        ur_se_total_6, ur_min_snr_6, ur_se_6 = data_process(np.reshape(H_c[5,episode_steps,:,ue_select_6],(num_bs,-1)),idx_6,mod_select_6)
        ur_se_total_6 = ur_se_total_6/normal_ur[5,episode_steps]

        mod_select_7 = np.ones((idx_7,)) *16 # 16-QAM
        ur_se_total_7, ur_min_snr_7, ur_se_7 = data_process(np.reshape(H_c[6,episode_steps,:,ue_select_7],(num_bs,-1)),idx_7,mod_select_7)
        ur_se_total_7 = ur_se_total_7/normal_ur[6,episode_steps]

        mod_select_8 = np.ones((idx_8,)) *16
        ur_se_total_8, ur_min_snr_8, ur_se_8 = data_process(np.reshape(H_c[7,episode_steps,:,ue_select_8],(num_bs,-1)),idx_8,mod_select_8)
        ur_se_total_8 = ur_se_total_8/normal_ur[7,episode_steps]

        mod_select_9 = np.ones((idx_9,)) *16 # 16-QAM
        ur_se_total_9, ur_min_snr_9, ur_se_9 = data_process(np.reshape(H_c[8,episode_steps,:,ue_select_9],(num_bs,-1)),idx_9,mod_select_9)
        ur_se_total_9 = ur_se_total_9/normal_ur[8,episode_steps]

        mod_select_10 = np.ones((idx_10,)) *16
        ur_se_total_10, ur_min_snr_10, ur_se_10 = data_process(np.reshape(H_c[9,episode_steps,:,ue_select_10],(num_bs,-1)),idx_10,mod_select_10)
        ur_se_total_10 = ur_se_total_10/normal_ur[9,episode_steps]


        mod_select_11 = np.ones((idx_11,)) *16 # 16-QAM
        ur_se_total_11, ur_min_snr_11, ur_se_11 = data_process(np.reshape(H_c[10,episode_steps,:,ue_select_11],(num_bs,-1)),idx_11,mod_select_11)
        ur_se_total_11 = ur_se_total_11/normal_ur[10,episode_steps]

        mod_select_12 = np.ones((idx_12,)) *16
        ur_se_total_12, ur_min_snr_12, ur_se_12 = data_process(np.reshape(H_c[11,episode_steps,:,ue_select_12],(num_bs,-1)),idx_12,mod_select_12)
        ur_se_total_12 = ur_se_total_12/normal_ur[11,episode_steps]

        mod_select_13 = np.ones((idx_13,)) *16 # 36-QAM
        ur_se_total_13, ur_min_snr_13, ur_se_13 = data_process(np.reshape(H_c[12,episode_steps,:,ue_select_13],(num_bs,-1)),idx_13,mod_select_13)
        ur_se_total_13 = ur_se_total_13/normal_ur[12,episode_steps]

        mod_select_14 = np.ones((idx_14,)) *16
        ur_se_total_14, ur_min_snr_14, ur_se_14 = data_process(np.reshape(H_c[13,episode_steps,:,ue_select_14],(num_bs,-1)),idx_14,mod_select_14)
        ur_se_total_14 = ur_se_total_14/normal_ur[13,episode_steps]

        mod_select_15 = np.ones((idx_15,)) *16 # 16-QAM
        ur_se_total_15, ur_min_snr_15, ur_se_15 = data_process(np.reshape(H_c[14,episode_steps,:,ue_select_15],(num_bs,-1)),idx_15,mod_select_15)
        ur_se_total_15 = ur_se_total_15/normal_ur[14,episode_steps]

        mod_select_16 = np.ones((idx_16,)) *16
        ur_se_total_16, ur_min_snr_16, ur_se_16 = data_process(np.reshape(H_c[15,episode_steps,:,ue_select_16],(num_bs,-1)),idx_16,mod_select_16)
        ur_se_total_16 = ur_se_total_16/normal_ur[15,episode_steps]

        mod_select_17 = np.ones((idx_17,)) *16 # 16-QAM
        ur_se_total_17, ur_min_snr_17, ur_se_17 = data_process(np.reshape(H_c[16,episode_steps,:,ue_select_17],(num_bs,-1)),idx_17,mod_select_17)
        ur_se_total_17 = ur_se_total_17/normal_ur[16,episode_steps]

        mod_select_18 = np.ones((idx_18,)) *16
        ur_se_total_18, ur_min_snr_18, ur_se_18 = data_process(np.reshape(H_c[17,episode_steps,:,ue_select_18],(num_bs,-1)),idx_18,mod_select_18)
        ur_se_total_18 = ur_se_total_18/normal_ur[17,episode_steps]

        mod_select_19 = np.ones((idx_19,)) *16 # 16-QAM
        ur_se_total_19, ur_min_snr_19, ur_se_19 = data_process(np.reshape(H_c[18,episode_steps,:,ue_select_19],(num_bs,-1)),idx_19,mod_select_19)
        ur_se_total_19 = ur_se_total_19/normal_ur[18,episode_steps]

        mod_select_20 = np.ones((idx_20,)) *16
        ur_se_total_20, ur_min_snr_20, ur_se_20 = data_process(np.reshape(H_c[19,episode_steps,:,ue_select_20],(num_bs,-1)),idx_20,mod_select_20)
        ur_se_total_20 = ur_se_total_20/normal_ur[19,episode_steps]


        mod_select_21 = np.ones((idx_21,)) *16 # 16-QAM
        ur_se_total_21, ur_min_snr_21, ur_se_21 = data_process(np.reshape(H_c[20,episode_steps,:,ue_select_21],(num_bs,-1)),idx_21,mod_select_21)
        ur_se_total_21 = ur_se_total_21/normal_ur[20,episode_steps]

        mod_select_22 = np.ones((idx_22,)) *16
        ur_se_total_22, ur_min_snr_22, ur_se_22 = data_process(np.reshape(H_c[21,episode_steps,:,ue_select_22],(num_bs,-1)),idx_22,mod_select_22)
        ur_se_total_22 = ur_se_total_22/normal_ur[21,episode_steps]

        mod_select_23 = np.ones((idx_23,)) *16 # 36-QAM
        ur_se_total_23, ur_min_snr_23, ur_se_23 = data_process(np.reshape(H_c[22,episode_steps,:,ue_select_23],(num_bs,-1)),idx_23,mod_select_23)
        ur_se_total_23 = ur_se_total_23/normal_ur[22,episode_steps]

        mod_select_24 = np.ones((idx_24,)) *16
        ur_se_total_24, ur_min_snr_24, ur_se_24 = data_process(np.reshape(H_c[23,episode_steps,:,ue_select_24],(num_bs,-1)),idx_24,mod_select_24)
        ur_se_total_24 = ur_se_total_24/normal_ur[23,episode_steps]

        mod_select_25 = np.ones((idx_25,)) *16 # 16-QAM
        ur_se_total_25, ur_min_snr_25, ur_se_25 = data_process(np.reshape(H_c[24,episode_steps,:,ue_select_25],(num_bs,-1)),idx_25,mod_select_25)
        ur_se_total_25 = ur_se_total_25/normal_ur[24,episode_steps]

        mod_select_26 = np.ones((idx_26,)) *16
        ur_se_total_26, ur_min_snr_26, ur_se_26 = data_process(np.reshape(H_c[25,episode_steps,:,ue_select_26],(num_bs,-1)),idx_26,mod_select_26)
        ur_se_total_26 = ur_se_total_26/normal_ur[25,episode_steps]

        mod_select_27 = np.ones((idx_27,)) *16 # 16-QAM
        ur_se_total_27, ur_min_snr_27, ur_se_27 = data_process(np.reshape(H_c[26,episode_steps,:,ue_select_27],(num_bs,-1)),idx_27,mod_select_27)
        ur_se_total_27 = ur_se_total_27/normal_ur[26,episode_steps]

        mod_select_28 = np.ones((idx_28,)) *16
        ur_se_total_28, ur_min_snr_28, ur_se_28 = data_process(np.reshape(H_c[27,episode_steps,:,ue_select_28],(num_bs,-1)),idx_28,mod_select_28)
        ur_se_total_28 = ur_se_total_28/normal_ur[27,episode_steps]

        mod_select_29 = np.ones((idx_29,)) *16 # 16-QAM
        ur_se_total_29, ur_min_snr_29, ur_se_29 = data_process(np.reshape(H_c[28,episode_steps,:,ue_select_29],(num_bs,-1)),idx_29,mod_select_29)
        ur_se_total_29 = ur_se_total_29/normal_ur[28,episode_steps]

        mod_select_30 = np.ones((idx_30,)) *16
        ur_se_total_30, ur_min_snr_30, ur_se_30 = data_process(np.reshape(H_c[29,episode_steps,:,ue_select_30],(num_bs,-1)),idx_30,mod_select_30)
        ur_se_total_30 = ur_se_total_30/normal_ur[29,episode_steps]


        mod_select_31 = np.ones((idx_31,)) *16 # 16-QAM
        ur_se_total_31, ur_min_snr_31, ur_se_31 = data_process(np.reshape(H_c[30,episode_steps,:,ue_select_31],(num_bs,-1)),idx_31,mod_select_31)
        ur_se_total_31 = ur_se_total_31/normal_ur[0,episode_steps]

        mod_select_32 = np.ones((idx_32,)) *16
        ur_se_total_32, ur_min_snr_32, ur_se_32 = data_process(np.reshape(H_c[31,episode_steps,:,ue_select_32],(num_bs,-1)),idx_32,mod_select_32)
        ur_se_total_32 = ur_se_total_32/normal_ur[31,episode_steps]

        mod_select_33 = np.ones((idx_33,)) *16 # 36-QAM
        ur_se_total_33, ur_min_snr_33, ur_se_33 = data_process(np.reshape(H_c[32,episode_steps,:,ue_select_33],(num_bs,-1)),idx_33,mod_select_33)
        ur_se_total_33 = ur_se_total_33/normal_ur[32,episode_steps]

        mod_select_34 = np.ones((idx_34,)) *16
        ur_se_total_34, ur_min_snr_34, ur_se_34 = data_process(np.reshape(H_c[33,episode_steps,:,ue_select_34],(num_bs,-1)),idx_34,mod_select_34)
        ur_se_total_34 = ur_se_total_34/normal_ur[33,episode_steps]

        mod_select_35 = np.ones((idx_35,)) *16 # 16-QAM
        ur_se_total_35, ur_min_snr_35, ur_se_35 = data_process(np.reshape(H_c[34,episode_steps,:,ue_select_35],(num_bs,-1)),idx_35,mod_select_35)
        ur_se_total_35 = ur_se_total_35/normal_ur[34,episode_steps]

        mod_select_36 = np.ones((idx_36,)) *16
        ur_se_total_36, ur_min_snr_36, ur_se_36 = data_process(np.reshape(H_c[35,episode_steps,:,ue_select_36],(num_bs,-1)),idx_36,mod_select_36)
        ur_se_total_36 = ur_se_total_36/normal_ur[35,episode_steps]

        mod_select_37 = np.ones((idx_37,)) *16 # 16-QAM
        ur_se_total_37, ur_min_snr_37, ur_se_37 = data_process(np.reshape(H_c[36,episode_steps,:,ue_select_37],(num_bs,-1)),idx_37,mod_select_37)
        ur_se_total_37 = ur_se_total_37/normal_ur[36,episode_steps]

        mod_select_38 = np.ones((idx_38,)) *16
        ur_se_total_38, ur_min_snr_38, ur_se_38 = data_process(np.reshape(H_c[37,episode_steps,:,ue_select_38],(num_bs,-1)),idx_38,mod_select_38)
        ur_se_total_38 = ur_se_total_38/normal_ur[37,episode_steps]

        mod_select_39 = np.ones((idx_39,)) *16 # 16-QAM
        ur_se_total_39, ur_min_snr_39, ur_se_39 = data_process(np.reshape(H_c[38,episode_steps,:,ue_select_39],(num_bs,-1)),idx_39,mod_select_39)
        ur_se_total_39 = ur_se_total_39/normal_ur[38,episode_steps]

        mod_select_40 = np.ones((idx_40,)) *16
        ur_se_total_40, ur_min_snr_40, ur_se_40 = data_process(np.reshape(H_c[39,episode_steps,:,ue_select_40],(num_bs,-1)),idx_40,mod_select_40)
        ur_se_total_40 = ur_se_total_40/normal_ur[39,episode_steps]



        mod_select_41 = np.ones((idx_41,)) *16 # 16-QAM
        ur_se_total_41, ur_min_snr_41, ur_se_41 = data_process(np.reshape(H_c[40,episode_steps,:,ue_select_41],(num_bs,-1)),idx_41,mod_select_41)
        ur_se_total_41 = ur_se_total_41/normal_ur[40,episode_steps]

        mod_select_42 = np.ones((idx_42,)) *16
        ur_se_total_42, ur_min_snr_42, ur_se_42 = data_process(np.reshape(H_c[41,episode_steps,:,ue_select_42],(num_bs,-1)),idx_42,mod_select_42)
        ur_se_total_42 = ur_se_total_42/normal_ur[41,episode_steps]

        mod_select_43 = np.ones((idx_43,)) *16 # 36-QAM
        ur_se_total_43, ur_min_snr_43, ur_se_43 = data_process(np.reshape(H_c[42,episode_steps,:,ue_select_43],(num_bs,-1)),idx_43,mod_select_43)
        ur_se_total_43 = ur_se_total_43/normal_ur[42,episode_steps]

        mod_select_44 = np.ones((idx_44,)) *16
        ur_se_total_44, ur_min_snr_44, ur_se_44 = data_process(np.reshape(H_c[43,episode_steps,:,ue_select_44],(num_bs,-1)),idx_44,mod_select_44)
        ur_se_total_44 = ur_se_total_44/normal_ur[43,episode_steps]

        mod_select_45 = np.ones((idx_45,)) *16 # 16-QAM
        ur_se_total_45, ur_min_snr_45, ur_se_45 = data_process(np.reshape(H_c[44,episode_steps,:,ue_select_45],(num_bs,-1)),idx_45,mod_select_45)
        ur_se_total_45 = ur_se_total_45/normal_ur[44,episode_steps]

        mod_select_46 = np.ones((idx_46,)) *16
        ur_se_total_46, ur_min_snr_46, ur_se_46 = data_process(np.reshape(H_c[45,episode_steps,:,ue_select_46],(num_bs,-1)),idx_46,mod_select_46)
        ur_se_total_46 = ur_se_total_46/normal_ur[45,episode_steps]

        mod_select_47 = np.ones((idx_47,)) *16 # 16-QAM
        ur_se_total_47, ur_min_snr_47, ur_se_47 = data_process(np.reshape(H_c[46,episode_steps,:,ue_select_47],(num_bs,-1)),idx_47,mod_select_47)
        ur_se_total_47 = ur_se_total_47/normal_ur[46,episode_steps]

        mod_select_48 = np.ones((idx_48,)) *16
        ur_se_total_48, ur_min_snr_48, ur_se_48 = data_process(np.reshape(H_c[47,episode_steps,:,ue_select_48],(num_bs,-1)),idx_48,mod_select_48)
        ur_se_total_48 = ur_se_total_48/normal_ur[47,episode_steps]

        mod_select_49 = np.ones((idx_49,)) *16 # 16-QAM
        ur_se_total_49, ur_min_snr_49, ur_se_49 = data_process(np.reshape(H_c[48,episode_steps,:,ue_select_49],(num_bs,-1)),idx_49,mod_select_49)
        ur_se_total_49 = ur_se_total_49/normal_ur[48,episode_steps]

        mod_select_50 = np.ones((idx_50,)) *16
        ur_se_total_50, ur_min_snr_50, ur_se_50 = data_process(np.reshape(H_c[49,episode_steps,:,ue_select_50],(num_bs,-1)),idx_50,mod_select_50)
        ur_se_total_50 = ur_se_total_50/normal_ur[49,episode_steps]

        for i in range(0,idx_1):
            ue_history[ue_select_1[i]] += ur_se_1[i]
        for i in range(0,idx_2):
            ue_history[ue_select_2[i]] += ur_se_2[i]
        for i in range(0,idx_3):
            ue_history[ue_select_3[i]] += ur_se_3[i]
        for i in range(0,idx_4):
            ue_history[ue_select_4[i]] += ur_se_4[i]
        for i in range(0,idx_5):
            ue_history[ue_select_5[i]] += ur_se_5[i]
        for i in range(0,idx_6):
            ue_history[ue_select_6[i]] += ur_se_6[i]
        for i in range(0,idx_7):
            ue_history[ue_select_7[i]] += ur_se_7[i]
        for i in range(0,idx_8):
            ue_history[ue_select_8[i]] += ur_se_8[i]
        for i in range(0,idx_9):
            ue_history[ue_select_9[i]] += ur_se_9[i]
        for i in range(0,idx_10):
            ue_history[ue_select_10[i]] += ur_se_10[i]

        for i in range(0,idx_11):
            ue_history[ue_select_11[i]] += ur_se_11[i]
        for i in range(0,idx_12):
            ue_history[ue_select_12[i]] += ur_se_12[i]
        for i in range(0,idx_13):
            ue_history[ue_select_13[i]] += ur_se_13[i]
        for i in range(0,idx_14):
            ue_history[ue_select_14[i]] += ur_se_14[i]
        for i in range(0,idx_15):
            ue_history[ue_select_15[i]] += ur_se_15[i]
        for i in range(0,idx_16):
            ue_history[ue_select_16[i]] += ur_se_16[i]
        for i in range(0,idx_17):
            ue_history[ue_select_17[i]] += ur_se_17[i]
        for i in range(0,idx_18):
            ue_history[ue_select_18[i]] += ur_se_18[i]
        for i in range(0,idx_19):
            ue_history[ue_select_19[i]] += ur_se_19[i]
        for i in range(0,idx_20):
            ue_history[ue_select_20[i]] += ur_se_20[i]

        for i in range(0,idx_21):
            ue_history[ue_select_21[i]] += ur_se_21[i]
        for i in range(0,idx_22):
            ue_history[ue_select_22[i]] += ur_se_22[i]
        for i in range(0,idx_23):
            ue_history[ue_select_23[i]] += ur_se_23[i]
        for i in range(0,idx_24):
            ue_history[ue_select_24[i]] += ur_se_24[i]
        for i in range(0,idx_25):
            ue_history[ue_select_25[i]] += ur_se_25[i]
        for i in range(0,idx_26):
            ue_history[ue_select_26[i]] += ur_se_26[i]
        for i in range(0,idx_27):
            ue_history[ue_select_27[i]] += ur_se_27[i]
        for i in range(0,idx_28):
            ue_history[ue_select_28[i]] += ur_se_28[i]
        for i in range(0,idx_29):
            ue_history[ue_select_29[i]] += ur_se_29[i]
        for i in range(0,idx_30):
            ue_history[ue_select_30[i]] += ur_se_30[i]


        for i in range(0,idx_31):
            ue_history[ue_select_31[i]] += ur_se_31[i]
        for i in range(0,idx_32):
            ue_history[ue_select_32[i]] += ur_se_32[i]
        for i in range(0,idx_33):
            ue_history[ue_select_33[i]] += ur_se_33[i]
        for i in range(0,idx_34):
            ue_history[ue_select_34[i]] += ur_se_34[i]
        for i in range(0,idx_35):
            ue_history[ue_select_35[i]] += ur_se_35[i]
        for i in range(0,idx_36):
            ue_history[ue_select_36[i]] += ur_se_36[i]
        for i in range(0,idx_37):
            ue_history[ue_select_37[i]] += ur_se_37[i]
        for i in range(0,idx_38):
            ue_history[ue_select_38[i]] += ur_se_38[i]
        for i in range(0,idx_39):
            ue_history[ue_select_39[i]] += ur_se_39[i]
        for i in range(0,idx_40):
            ue_history[ue_select_40[i]] += ur_se_40[i]


        for i in range(0,idx_41):
            ue_history[ue_select_41[i]] += ur_se_41[i]
        for i in range(0,idx_42):
            ue_history[ue_select_42[i]] += ur_se_42[i]
        for i in range(0,idx_43):
            ue_history[ue_select_43[i]] += ur_se_43[i]
        for i in range(0,idx_44):
            ue_history[ue_select_44[i]] += ur_se_44[i]
        for i in range(0,idx_45):
            ue_history[ue_select_45[i]] += ur_se_45[i]
        for i in range(0,idx_46):
            ue_history[ue_select_46[i]] += ur_se_46[i]
        for i in range(0,idx_47):
            ue_history[ue_select_47[i]] += ur_se_47[i]
        for i in range(0,idx_48):
            ue_history[ue_select_48[i]] += ur_se_48[i]
        for i in range(0,idx_49):
            ue_history[ue_select_49[i]] += ur_se_49[i]
        for i in range(0,idx_50):
            ue_history[ue_select_50[i]] += ur_se_50[i]

        history_record[episode_steps,:] = ue_history

        # print("history record:",history_record[episode_steps,:])

        jfi= np.square((np.sum(ue_history))) / (num_ue * np.sum(np.square(ue_history)))
       
        # jfi_1 = np.square((np.sum(ue_history_1))) / (num_ue * np.sum(np.square(ue_history_1)))
        # jfi_2 = np.square((np.sum(ue_history_2))) / (num_ue * np.sum(np.square(ue_history_2)))
        # print('jfi is', jfi_1)
        # next_state_1 = np.concatenate((np.reshape(se_max_ur_1[(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history_1,(1,num_ue)),np.reshape(H_r_1[(episode_steps+1),:,:],(1,-1)),np.reshape(H_i_1[(episode_steps+1),:,:],(1,-1))),axis = 1)
        # next_state_2 = np.concatenate((np.reshape(se_max_ur_2[(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history_1,(1,num_ue)),np.reshape(H_r_2[(episode_steps+1),:,:],(1,-1)),np.reshape(H_i_2[(episode_steps+1),:,:],(1,-1))),axis = 1)
        
        next_state_1 = np.concatenate((np.reshape(se_max_ur[0,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[0,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[0,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_2 = np.concatenate((np.reshape(se_max_ur[1,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[1,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[1,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_3 = np.concatenate((np.reshape(se_max_ur[2,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[2,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[2,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_4 = np.concatenate((np.reshape(se_max_ur[3,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[3,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[3,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_5 = np.concatenate((np.reshape(se_max_ur[4,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[4,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[4,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_6 = np.concatenate((np.reshape(se_max_ur[5,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[5,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[5,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_7 = np.concatenate((np.reshape(se_max_ur[6,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[6,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[6,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_8 = np.concatenate((np.reshape(se_max_ur[7,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[7,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[7,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_9 = np.concatenate((np.reshape(se_max_ur[8,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[8,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[8,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_10 = np.concatenate((np.reshape(se_max_ur[9,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[9,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[9,(episode_steps+1),:,:],(1,-1))),axis = 1)

        next_state_11 = np.concatenate((np.reshape(se_max_ur[10,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[10,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[10,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_12 = np.concatenate((np.reshape(se_max_ur[11,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[11,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[11,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_13 = np.concatenate((np.reshape(se_max_ur[12,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[12,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[12,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_14 = np.concatenate((np.reshape(se_max_ur[13,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[13,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[13,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_15 = np.concatenate((np.reshape(se_max_ur[14,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[14,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[14,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_16 = np.concatenate((np.reshape(se_max_ur[15,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[15,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[15,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_17 = np.concatenate((np.reshape(se_max_ur[16,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[16,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[16,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_18 = np.concatenate((np.reshape(se_max_ur[17,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[17,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[17,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_19 = np.concatenate((np.reshape(se_max_ur[18,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[18,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[18,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_20 = np.concatenate((np.reshape(se_max_ur[19,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[19,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[19,(episode_steps+1),:,:],(1,-1))),axis = 1)

        next_state_21 = np.concatenate((np.reshape(se_max_ur[20,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[20,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[20,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_22 = np.concatenate((np.reshape(se_max_ur[21,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[21,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[21,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_23 = np.concatenate((np.reshape(se_max_ur[22,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[22,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[22,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_24 = np.concatenate((np.reshape(se_max_ur[23,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[23,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[23,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_25 = np.concatenate((np.reshape(se_max_ur[24,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[24,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[24,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_26 = np.concatenate((np.reshape(se_max_ur[25,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[25,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[25,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_27 = np.concatenate((np.reshape(se_max_ur[26,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[26,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[26,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_28 = np.concatenate((np.reshape(se_max_ur[27,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[27,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[27,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_29 = np.concatenate((np.reshape(se_max_ur[28,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[28,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[28,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_30 = np.concatenate((np.reshape(se_max_ur[29,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[29,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[29,(episode_steps+1),:,:],(1,-1))),axis = 1)

        next_state_31 = np.concatenate((np.reshape(se_max_ur[30,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[30,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[30,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_32 = np.concatenate((np.reshape(se_max_ur[31,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[31,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[31,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_33 = np.concatenate((np.reshape(se_max_ur[32,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[32,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[32,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_34 = np.concatenate((np.reshape(se_max_ur[33,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[33,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[33,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_35 = np.concatenate((np.reshape(se_max_ur[34,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[34,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[34,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_36 = np.concatenate((np.reshape(se_max_ur[35,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[35,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[35,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_37 = np.concatenate((np.reshape(se_max_ur[36,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[36,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[36,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_38 = np.concatenate((np.reshape(se_max_ur[37,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[37,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[37,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_39 = np.concatenate((np.reshape(se_max_ur[38,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[38,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[38,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_40 = np.concatenate((np.reshape(se_max_ur[39,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[39,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[39,(episode_steps+1),:,:],(1,-1))),axis = 1)

        next_state_41 = np.concatenate((np.reshape(se_max_ur[40,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[40,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[40,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_42 = np.concatenate((np.reshape(se_max_ur[41,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[41,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[41,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_43 = np.concatenate((np.reshape(se_max_ur[42,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[42,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[42,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_44 = np.concatenate((np.reshape(se_max_ur[43,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[43,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[43,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_45 = np.concatenate((np.reshape(se_max_ur[44,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[44,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[44,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_46 = np.concatenate((np.reshape(se_max_ur[45,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[45,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[45,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_47 = np.concatenate((np.reshape(se_max_ur[46,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[46,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[46,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_48 = np.concatenate((np.reshape(se_max_ur[47,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[47,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[47,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_49 = np.concatenate((np.reshape(se_max_ur[48,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[48,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[48,(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_50 = np.concatenate((np.reshape(se_max_ur[49,(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history,(1,num_ue)),np.reshape(H_rc[49,(episode_steps+1),:,:],(1,-1)),np.reshape(H_ic[49,(episode_steps+1),:,:],(1,-1))),axis = 1)
        
        reward_1  = (a*ur_se_total_1 + b*jfi)*reward_scale
        reward_2  = (a*ur_se_total_2 + b*jfi)*reward_scale
        reward_3  = (a*ur_se_total_3 + b*jfi)*reward_scale
        reward_4  = (a*ur_se_total_4 + b*jfi)*reward_scale
        reward_5  = (a*ur_se_total_5 + b*jfi)*reward_scale
        reward_6  = (a*ur_se_total_6 + b*jfi)*reward_scale
        reward_7  = (a*ur_se_total_7 + b*jfi)*reward_scale
        reward_8  = (a*ur_se_total_8 + b*jfi)*reward_scale
        reward_9  = (a*ur_se_total_9 + b*jfi)*reward_scale
        reward_10  = (a*ur_se_total_10 + b*jfi)*reward_scale

        reward_11  = (a*ur_se_total_11 + b*jfi)*reward_scale
        reward_12  = (a*ur_se_total_12 + b*jfi)*reward_scale
        reward_13  = (a*ur_se_total_13 + b*jfi)*reward_scale
        reward_14  = (a*ur_se_total_14 + b*jfi)*reward_scale
        reward_15  = (a*ur_se_total_15 + b*jfi)*reward_scale
        reward_16  = (a*ur_se_total_16 + b*jfi)*reward_scale
        reward_17  = (a*ur_se_total_17 + b*jfi)*reward_scale
        reward_18  = (a*ur_se_total_18 + b*jfi)*reward_scale
        reward_19  = (a*ur_se_total_19 + b*jfi)*reward_scale
        reward_20  = (a*ur_se_total_20 + b*jfi)*reward_scale

        reward_21  = (a*ur_se_total_21 + b*jfi)*reward_scale
        reward_22  = (a*ur_se_total_22 + b*jfi)*reward_scale
        reward_23  = (a*ur_se_total_23 + b*jfi)*reward_scale
        reward_24  = (a*ur_se_total_24 + b*jfi)*reward_scale
        reward_25  = (a*ur_se_total_25 + b*jfi)*reward_scale
        reward_26  = (a*ur_se_total_26 + b*jfi)*reward_scale
        reward_27  = (a*ur_se_total_27 + b*jfi)*reward_scale
        reward_28  = (a*ur_se_total_28 + b*jfi)*reward_scale
        reward_29  = (a*ur_se_total_29 + b*jfi)*reward_scale
        reward_30  = (a*ur_se_total_30 + b*jfi)*reward_scale

        reward_31  = (a*ur_se_total_31 + b*jfi)*reward_scale
        reward_32  = (a*ur_se_total_32 + b*jfi)*reward_scale
        reward_33  = (a*ur_se_total_33 + b*jfi)*reward_scale
        reward_34  = (a*ur_se_total_34 + b*jfi)*reward_scale
        reward_35  = (a*ur_se_total_35 + b*jfi)*reward_scale
        reward_36  = (a*ur_se_total_36 + b*jfi)*reward_scale
        reward_37  = (a*ur_se_total_37 + b*jfi)*reward_scale
        reward_38  = (a*ur_se_total_38 + b*jfi)*reward_scale
        reward_39  = (a*ur_se_total_39 + b*jfi)*reward_scale
        reward_40  = (a*ur_se_total_40 + b*jfi)*reward_scale

        reward_41  = (a*ur_se_total_41 + b*jfi)*reward_scale
        reward_42  = (a*ur_se_total_42 + b*jfi)*reward_scale
        reward_43  = (a*ur_se_total_43 + b*jfi)*reward_scale
        reward_44  = (a*ur_se_total_44 + b*jfi)*reward_scale
        reward_45  = (a*ur_se_total_45 + b*jfi)*reward_scale
        reward_46  = (a*ur_se_total_46 + b*jfi)*reward_scale
        reward_47  = (a*ur_se_total_47 + b*jfi)*reward_scale
        reward_48  = (a*ur_se_total_48 + b*jfi)*reward_scale
        reward_49  = (a*ur_se_total_49 + b*jfi)*reward_scale
        reward_50  = (a*ur_se_total_50 + b*jfi)*reward_scale


        done = False
        # print('reward is:', reward_1,reward_2)
        if args.max_episode_steps and episode_steps >= args.max_episode_steps-2:
            done = True
            

        episode_steps += 1
        total_numsteps += 1
        # episode_reward_1 += reward_1
        # episode_reward_2 += reward_2
        # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
        # print('episode reward is:', episode_reward_1, episode_reward_2,'\n')
        
        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps >= args.max_episode_steps -1 else float(not done)
        # print('mask is:',mask)

        memory_1.push(state_1, action_1, reward_1, next_state_1, mask) # Append transition to memory
        memory_2.push(state_2, action_2, reward_2, next_state_2, mask)
        memory_3.push(state_3, action_3, reward_3, next_state_3, mask) # Append transition to memory
        memory_4.push(state_4, action_4, reward_4, next_state_4, mask)
        memory_5.push(state_5, action_5, reward_5, next_state_5, mask) # Append transition to memory
        memory_6.push(state_6, action_6, reward_6, next_state_6, mask)
        memory_7.push(state_7, action_7, reward_7, next_state_7, mask) # Append transition to memory
        memory_8.push(state_8, action_8, reward_8, next_state_8, mask)
        memory_9.push(state_9, action_9, reward_9, next_state_9, mask) # Append transition to memory
        memory_10.push(state_10, action_10, reward_10, next_state_10, mask)

        memory_11.push(state_11, action_11, reward_11, next_state_11, mask) # Append transition to memory
        memory_12.push(state_12, action_12, reward_12, next_state_12, mask)
        memory_13.push(state_13, action_13, reward_13, next_state_13, mask) # Append transition to memory
        memory_14.push(state_14, action_14, reward_14, next_state_14, mask)
        memory_15.push(state_15, action_15, reward_15, next_state_15, mask) # Append transition to memory
        memory_16.push(state_16, action_16, reward_16, next_state_16, mask)
        memory_17.push(state_17, action_17, reward_17, next_state_17, mask) # Append transition to memory
        memory_18.push(state_18, action_18, reward_18, next_state_18, mask)
        memory_19.push(state_19, action_19, reward_19, next_state_19, mask) # Append transition to memory
        memory_20.push(state_20, action_20, reward_20, next_state_20, mask)

        memory_21.push(state_21, action_21, reward_21, next_state_21, mask) # Append transition to memory
        memory_22.push(state_22, action_22, reward_22, next_state_22, mask)
        memory_23.push(state_23, action_23, reward_23, next_state_23, mask) # Append transition to memory
        memory_24.push(state_24, action_24, reward_24, next_state_24, mask)
        memory_25.push(state_25, action_25, reward_25, next_state_25, mask) # Append transition to memory
        memory_26.push(state_26, action_26, reward_26, next_state_26, mask)
        memory_27.push(state_27, action_27, reward_27, next_state_27, mask) # Append transition to memory
        memory_28.push(state_28, action_28, reward_28, next_state_28, mask)
        memory_29.push(state_29, action_29, reward_29, next_state_29, mask) # Append transition to memory
        memory_30.push(state_30, action_30, reward_30, next_state_30, mask)

        memory_31.push(state_31, action_31, reward_31, next_state_31, mask) # Append transition to memory
        memory_32.push(state_32, action_32, reward_32, next_state_32, mask)
        memory_33.push(state_33, action_33, reward_33, next_state_33, mask) # Append transition to memory
        memory_34.push(state_34, action_34, reward_34, next_state_34, mask)
        memory_35.push(state_35, action_35, reward_35, next_state_35, mask) # Append transition to memory
        memory_36.push(state_36, action_36, reward_36, next_state_36, mask)
        memory_37.push(state_37, action_37, reward_37, next_state_37, mask) # Append transition to memory
        memory_38.push(state_38, action_38, reward_38, next_state_38, mask)
        memory_39.push(state_39, action_39, reward_39, next_state_39, mask) # Append transition to memory
        memory_40.push(state_40, action_40, reward_40, next_state_40, mask)

        memory_41.push(state_41, action_41, reward_41, next_state_41, mask) # Append transition to memory
        memory_42.push(state_42, action_42, reward_42, next_state_42, mask)
        memory_43.push(state_43, action_43, reward_43, next_state_43, mask) # Append transition to memory
        memory_44.push(state_44, action_44, reward_44, next_state_44, mask)
        memory_45.push(state_45, action_45, reward_45, next_state_45, mask) # Append transition to memory
        memory_46.push(state_46, action_46, reward_46, next_state_46, mask)
        memory_47.push(state_47, action_47, reward_47, next_state_47, mask) # Append transition to memory
        memory_48.push(state_48, action_48, reward_48, next_state_48, mask)
        memory_49.push(state_49, action_49, reward_49, next_state_49, mask) # Append transition to memory
        memory_50.push(state_50, action_50, reward_50, next_state_50, mask)

        state_1 = next_state_1
        state_2 = next_state_2
        state_3 = next_state_3
        state_4 = next_state_4
        state_5 = next_state_5
        state_6 = next_state_6
        state_7 = next_state_7
        state_8 = next_state_8
        state_9 = next_state_9
        state_10 = next_state_10

        state_11 = next_state_11
        state_12 = next_state_12
        state_13 = next_state_13
        state_14 = next_state_14
        state_15 = next_state_15
        state_16 = next_state_16
        state_17 = next_state_17
        state_18 = next_state_18
        state_19 = next_state_19
        state_20 = next_state_20

        state_21 = next_state_21
        state_22 = next_state_22
        state_23 = next_state_23
        state_24 = next_state_24
        state_25 = next_state_25
        state_26 = next_state_26
        state_27 = next_state_27
        state_28 = next_state_28
        state_29 = next_state_29
        state_30 = next_state_30

        state_31 = next_state_31
        state_32 = next_state_32
        state_33 = next_state_33
        state_34 = next_state_34
        state_35 = next_state_35
        state_36 = next_state_36
        state_37 = next_state_37
        state_38 = next_state_38
        state_39 = next_state_39
        state_40 = next_state_40

        state_41 = next_state_41
        state_42 = next_state_42
        state_43 = next_state_43
        state_44 = next_state_44
        state_45 = next_state_45
        state_46 = next_state_46
        state_47 = next_state_47
        state_48 = next_state_48
        state_49 = next_state_49
        state_50 = next_state_50


        end_time = time.time()
        print('Training time is:', (end_time-start_time),"\n")

with h5py.File("./data/8_8_infer_50RB_static_mul_jfi.hdf5", "w") as data_file:

    data_file.create_dataset("history", data=history_record)

print('Training is finished\n')
# end_time = time.time()
# print('Training time is:', (end_time-start_time))