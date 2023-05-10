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

H_r_1 = H_rc[0,:,:,:]
H_i_1 = H_ic[0,:,:,:]

H_1 = np.array(H_r_1 + 1j*H_i_1)
print("H_1 shape is:", H_1.shape)

H_r_2 = H_ic[2,:,:,:]
H_i_2 = H_ic[2,:,:,:]

H_2 = np.array(H_r_2 + 1j*H_i_2)
print("H_2 shape is:", H_2.shape)

se_max_ur_1 = se_max_ur[2,:,:]
print("se_max_ur_1 shape is:", se_max_ur_1.shape)

se_max_ur_2 = se_max_ur[1,:,:]
print("se_max_ur_2 shape is:", se_max_ur_2.shape)

normal_ur_1 = normal_ur[2,:]
print("normal_ur_1 shape is:", normal_ur_1.shape)

normal_ur_2 = normal_ur[1,:]
print("normal_ur_2 shape is:", normal_ur_2.shape)
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

#Tensorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory_1 = ReplayMemory(args.replay_size, args.seed)
memory_2 = ReplayMemory(args.replay_size, args.seed)

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
ue_history_1 = 0.01*np.ones((num_ue,)) ## Tp history for each UE [4,]
ue_history_2 = 0.01*np.ones((num_ue,))

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
    episode_reward_1 = 0
    episode_reward_2 = 0
    episode_steps = 0
    done = False
    # ue_ep_history = np.zeros((400,num_ue))
    # ue_history = 0.01*np.ones((num_ue,))
    # ckpt_path_1 = "checkpoints/sac_checkpoint_16_4_16_vanilla_2RBmob_sep_"
    ckpt_path_1 = "checkpoints/sac_checkpoint_8_8_2RB_static_2_new_"
    ckpt_path_2 = "checkpoints/sac_checkpoint_8_8_2RB_static_2_new_"
    agent_1.load_checkpoint(ckpt_path_1)
    agent_2.load_checkpoint(ckpt_path_2)

    state_1 = np.concatenate((np.reshape(se_max_ur_1[0,:],(1,num_ue)),np.reshape(ue_history_1,(1,num_ue)),np.reshape(H_r_1[0,:,:],(1,-1)),np.reshape(H_i_1[0,:,:],(1,-1))),axis = 1)
    state_2 = np.concatenate((np.reshape(se_max_ur_2[0,:],(1,num_ue)),np.reshape(ue_history_1,(1,num_ue)),np.reshape(H_r_2[0,:,:],(1,-1)),np.reshape(H_i_2[0,:,:],(1,-1))),axis = 1)
    print('state shape is:',state_1.shape, state_2.shape)
    while not done:
        print('Training processing: ',i_episode,' episode ',episode_steps,' episode_steps')
        start_time = time.time()
        if random > np.random.rand(1):
            # if step <= warmup or (episode < 100):
            action_1, final_action_1 = agent_1.random_action()
            action_2, final_action_2 = agent_2.random_action()
            print('Random action',final_action_1, final_action_2)
        else:
            action_1, final_action_1 = agent_1.select_action(state_1)
            action_2, final_action_2 = agent_2.select_action(state_2)
            print('Actor action',final_action_1, final_action_2)
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
        # ur_se_total = pre_data[episode_steps,final_action[0],0]
        # ur_min_snr = pre_data[episode_steps,final_action[0],1]
        # ur_se = pre_data[episode_steps,final_action[0],2:2+idx]
        mod_select_1 = np.ones((idx_1,)) *16 # 16-QAM
        ur_se_total_1, ur_min_snr_1, ur_se_1 = data_process(np.reshape(H_1[episode_steps,:,ue_select_1],(num_bs,-1)),idx_1,mod_select_1)
        ur_se_total_1 = ur_se_total_1/normal_ur_1[episode_steps]

        mod_select_2 = np.ones((idx_2,)) *16 # 16-QAM
        ur_se_total_2, ur_min_snr_2, ur_se_2 = data_process(np.reshape(H_2[episode_steps,:,ue_select_2],(num_bs,-1)),idx_2,mod_select_2)
        ur_se_total_2 = ur_se_total_2/normal_ur_2[episode_steps]
        # print('se_total, min_snr are',ur_se_total_1, ur_se_total_2, ur_min_snr_1, ur_min_snr_2)
        # print('idx is',idx_1,idx_2)
        # print('ue_selection is:', ue_select_1,ue_select_2)
        for i in range(0,idx_1):
            ue_history_1[ue_select_1[i]] += ur_se_1[i]
        for i in range(0,idx_2):
            ue_history_1[ue_select_2[i]] += ur_se_2[i]
            # print('ur_se is',ur_se[i])
        
        # ue_ep_history[episode_steps,:] = ue_history

 
        history_record[episode_steps,:] = ue_history_1
        print("history record:",history_record[episode_steps,:])
       
        jfi_1 = np.square((np.sum(ue_history_1))) / (num_ue * np.sum(np.square(ue_history_1)))
        jfi_2 = np.square((np.sum(ue_history_2))) / (num_ue * np.sum(np.square(ue_history_2)))
        # print('jfi is', jfi_1)
        next_state_1 = np.concatenate((np.reshape(se_max_ur_1[(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history_1,(1,num_ue)),np.reshape(H_r_1[(episode_steps+1),:,:],(1,-1)),np.reshape(H_i_1[(episode_steps+1),:,:],(1,-1))),axis = 1)
        next_state_2 = np.concatenate((np.reshape(se_max_ur_2[(episode_steps+1),:],(1,num_ue)),np.reshape(ue_history_1,(1,num_ue)),np.reshape(H_r_2[(episode_steps+1),:,:],(1,-1)),np.reshape(H_i_2[(episode_steps+1),:,:],(1,-1))),axis = 1)
        reward_1  = (a*ur_se_total_1 + b*jfi_1 + c*ur_min_snr_1)*reward_scale
        reward_2  = (a*ur_se_total_2 + b*jfi_1 + c*ur_min_snr_2)*reward_scale
        done = False
        # print('reward is:', reward_1,reward_2)
        if args.max_episode_steps and episode_steps >= args.max_episode_steps-2:
            done = True
            

        # if len(memory_1) > args.batch_size:
        #     # Number of updates per step in environment
        #     for i in range(args.updates_per_step):
        #         # Update parameters of all the networks
        #         critic_1_loss_1, critic_2_loss_1, policy_loss_1, ent_loss_1, alpha_1 = agent_1.update_parameters(memory_1, args.batch_size, updates)

        #         writer.add_scalar('loss/critic_1_1', critic_1_loss_1, updates)
        #         writer.add_scalar('loss/critic_2_1', critic_2_loss_1, updates)
        #         writer.add_scalar('loss/policy_1', policy_loss_1, updates)
        #         writer.add_scalar('loss/entropy_loss_1', ent_loss_1, updates)
        #         writer.add_scalar('entropy_temprature/alpha_1', alpha_1, updates)
        #         updates += 1

        # if len(memory_2) > args.batch_size:
        #     # Number of updates per step in environment
        #     for i in range(args.updates_per_step):
        #         # Update parameters of all the networks
        #         critic_1_loss_2, critic_2_loss_2, policy_loss_2, ent_loss_2, alpha_2 = agent_2.update_parameters(memory_2, args.batch_size, updates)

        #         writer.add_scalar('loss/critic_1_2', critic_1_loss_2, updates)
        #         writer.add_scalar('loss/critic_2_2', critic_2_loss_2, updates)
        #         writer.add_scalar('loss/policy_2', policy_loss_2, updates)
        #         writer.add_scalar('loss/entropy_loss_2', ent_loss_2, updates)
        #         writer.add_scalar('entropy_temprature/alpha_2', alpha_2, updates)
        #         updates += 1
        
        # next_state, reward, done, _ = env.step(final_action) # Modify !!!!!!!!!!!!!!!!!
        episode_steps += 1
        total_numsteps += 1
        episode_reward_1 += reward_1
        episode_reward_2 += reward_2
        # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
        # print('episode reward is:', episode_reward_1, episode_reward_2,'\n')
        
        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps >= args.max_episode_steps -1 else float(not done)
        # print('mask is:',mask)

        memory_1.push(state_1, action_1, reward_1, next_state_1, mask) # Append transition to memory
        memory_2.push(state_2, action_2, reward_2, next_state_2, mask)

        state_1 = next_state_1
        state_2 = next_state_2
        end_time = time.time()
        print('Training time is:', (end_time-start_time),"\n")

    # if total_numsteps > args.num_steps:
    #     break

    writer.add_scalar('reward/train_1', episode_reward_1, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward_1: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward_1, 2)))
    writer.add_scalar('reward/train_2', episode_reward_2, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward_2: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward_2, 2)))

    # reward_record_1[i_episode] = episode_reward_1
    # reward_record_2[i_episode] = episode_reward_2
    # if episode_reward_1 == np.max(reward_record_1):
    #     max_history_record = ue_ep_history

    # if args.start_steps < total_numsteps and i_episode > 0 and i_episode % args.save_per_epochs == 0:
    #     agent_1.save_checkpoint('16_4_16_vanilla_2RBmob_1')
    #     agent_2.save_checkpoint('16_4_16_vanilla_2RBmob_2')


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


with h5py.File("./data/8_8_infer_2RBstatic_m.hdf5", "w") as data_file:
    # data_file.create_dataset("reward_1", data=reward_record_1)
    # data_file.create_dataset("reward_2", data=reward_record_2)
    data_file.create_dataset("history", data=history_record)
    # data_file.create_dataset("max_history", data=max_history_record)

print('Training is finished\n')
# end_time = time.time()
# print('Training time is:', (end_time-start_time))