import argparse
import datetime
import time
import numpy as np
from itertools import combinations
from collections.abc import Sequence
import torch
import h5py
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from mimo_data_process import *
from usr_group import *
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
parser.add_argument('--lr', type=float, default=0.0005, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha_lr', type=float, default=0.0005, metavar='G',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: True)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
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
args = parser.parse_args()

# Environment

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)



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


H_file = h5py.File('../6416_channel.hdf5','r')
H_r = np.array(H_file.get('H_r'))
H_i = np.array(H_file.get('H_i'))
H_i = np.transpose(H_i,(2,1,0))
H_r = np.transpose(H_r,(2,1,0))
print("H_r shape is:", H_r.shape)
print("H_i shape is:", H_i.shape)
H = np.array((H_r + 1j*H_i)[0:400,:,:])
print("H shape is:", H.shape)


# Import se_max

se_max_ur_file  = h5py.File('../6416_se_max.hdf5', 'r')
se_max_ur = se_max_ur_file.get('se_max')
se_max_ur = np.array(se_max_ur[0:400,:])
print("se_max_ur shape is:", se_max_ur.shape)


# H_file = h5py.File('../1_out_3_all.hdf5','r')
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

#############################################################

max_actions = 65536  #4294967295 = 65536^2 -1
num_states = 48 ### 16UE x (SE + history + UG_label)
num_actions = 2
random = 1
epsilon = 200000

# Agent
agent = SAC(num_states, num_actions, max_actions, args)

print('SAC build finished')

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
mod_set = [4,16,64]

### Reward Function Parameters
a = 2
b = 0.5
c = 0.01
reward_scale = 0.001
### Record vector defination
reward_record = np.zeros((800,))
history_record = np.zeros((10,400,16))
ue_history = 0.01*np.ones((16,)) ## Tp history for each UE [16,]

# def sel_ue(action):
#     sum_before = 0
#     for i in range (1,5):
#         sum_before += len(list(combinations(user_set, i)))
#         if ((action+1)>sum_before):
#             continue
#         else:
#             idx = i
#             sum_before -= len(list(combinations(user_set, i)))
#             ue_select = list(combinations(user_set, i))[action-sum_before]
#             break
#     return ue_select,idx

def sel_ue(action):
    sum_before = 0
    for i in range (1,17):
        sum_before += (3**i) * len(list(combinations(user_set, i)))
        if ((action+1)>sum_before):
            continue
        else:
            idx = i
            sum_before -= (3**i) * len(list(combinations(user_set, i)))
            break
    return idx,sum_before

def sel_mod (action,idx,sum_before):
    ue_remain = action  - sum_before
    num_1 = ue_remain // (3**idx) ## ue order
    # print('num_1:',num_1)
    ue_select = list(combinations(user_set, idx))
    ue_select = np.array(ue_select[num_1])
    mod_remain = ue_remain - num_1*(3**idx)
    mod_select = product_item(mod_remain, mod_set, repeat=idx)
    mod_select = np.array(mod_select)
    return ue_select,mod_select

def product_item(idx, *seqs, repeat=None):
    # Ensure inputs are actual sequences (list, tuple, str...)
    seqs = [seq if isinstance(seq, Sequence) else list(seq) for seq in seqs]
    # Repeat if needed
    if repeat is not None:
        seqs = seqs * repeat
    # Compute how many items does it take to advance on each sequence
    step = 1
    for seq in seqs:
        step *= len(seq)
    # Build product item
    item = [None] * len(seqs)
    for i, seq in enumerate(seqs):
        step //= len(seq)
        seq_idx = idx // step
        idx %= step
        item[i] = seq[seq_idx]
    return list(item)

start_time = time.time()
for i_episode in range (args.max_episode): 
    episode_reward = 0
    episode_steps = 0
    done = False
    ue_history = 0.01*np.ones((16,))
    group_idx = usr_group(np.squeeze(H[0,:,:]))
    state = np.concatenate((np.reshape(se_max_ur[0,:],(1,16)),np.reshape(ue_history,(1,16)),np.reshape(group_idx,(1,-1))),axis = 1)
    # print('state shape is:',state.shape)
    while not done:
        print('Training processing: ',i_episode,' episode ',episode_steps,' episode_steps')
        
        if random > np.random.rand(1):
            # if step <= warmup or (episode < 100):
            action, final_action = agent.random_action()
            print('Random action',final_action)
        else:
            action, final_action = agent.select_action(state)
            print('Actor action',final_action)
        # print('final action is: ', final_action)
        random -= 1/epsilon
        
        # if args.start_steps > total_numsteps:
        #     action, final_action = agent.random_action()
        #     # action = env.action_space.sample()  # # Modify !!!!!!!!!!!!!!!!!
        # else:
        #     action, final_action = agent.select_action(state)  # Sample action from policy
        # print('final action is: ', final_action)
        # print('final action is', final_action[0])
        idx,sum_before = sel_ue(final_action[0])
        ue_select,mod_select = sel_mod(final_action[0],idx,sum_before)

        # ue_select,idx = sel_ue(final_action[0])
        # mod_select = np.ones((idx,)) *16 # 16-QAM

        ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(H[episode_steps,:,ue_select],(64,-1)),idx,mod_select)
        print('se_total, min_snr are',ur_se_total, ur_min_snr)
        # print('idx is',idx)
        print('ue_selection and mod_selection are:', ue_select, mod_select)
        for i in range(0,idx):
            ue_history[ue_select[i]] += ur_se[i]
            # print('ur_se is',ur_se[i])
        
        if (i_episode >= 790):
            history_record[i_episode-790,episode_steps,:] = ue_history
       
        jfi = np.square((np.sum(ue_history))) / (16 * np.sum(np.square(ue_history)))
        print('jfi is', jfi)

        group_idx_next = usr_group(np.squeeze(H[(episode_steps+1),:,:]))
        next_state = np.concatenate((np.reshape(se_max_ur[(episode_steps+1),:],(1,16)),np.reshape(ue_history,(1,16)),np.reshape(group_idx_next,(1,-1))),axis = 1)
        
        reward  = (a*ur_se_total + b*jfi + c*ur_min_snr)*reward_scale
        done = False
        print('reward is:', reward)
        if args.max_episode_steps and episode_steps >= args.max_episode_steps-2:
            done = True
            

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
        # print('mask is:',mask)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    # if total_numsteps > args.num_steps:
    #     break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    reward_record[i_episode] = episode_reward

    if args.start_steps < total_numsteps and i_episode > 0 and i_episode % args.save_per_epochs == 0:
        agent.save_checkpoint('QuaDriGa_SAC_KNN_6416')


    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")


with h5py.File("./plot_data/plotdata_shuffled_6416.hdf5", "w") as data_file:
    data_file.create_dataset("reward", data=reward_record)
    data_file.create_dataset("history", data=history_record)

print('Training is finished\n')
end_time = time.time()
print('Training time is:', (end_time-start_time))


