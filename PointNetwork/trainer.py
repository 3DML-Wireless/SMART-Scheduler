#!/usr/bin/env python
import datetime
import argparse
import os
import time
from tqdm import tqdm 
import h5py
import pprint as pp
import numpy as np

import torch
print(torch.__version__)
import torch.optim as optim
import torch.autograd as autograd
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard_logger import configure
from neural_combinatorial_rl import NeuralCombOptRL
from mimo_sim_ul import *
from usr_group import *

def str2bool(v):
      return v.lower() in ('true', '1')

parser = argparse.ArgumentParser(description="Neural Combinatorial Optimization with RL")

# Data
parser.add_argument('--task', default='mob_64_16_4', help="The task to solve, in the form {COP}_{size}, e.g., mob_8_8_4")
parser.add_argument('--batch_size', default=128, help='')
parser.add_argument('--train_size', default=1000000, help='')
parser.add_argument('--val_size', default=10000, help='')
# Network
parser.add_argument('--embedding_dim', default=128, help='Dimension of input embedding')
parser.add_argument('--hidden_dim', default=128, help='Dimension of hidden layers in Enc/Dec')
parser.add_argument('--n_process_blocks', default=3, help='Number of process block iters to run in the Critic network')
parser.add_argument('--n_glimpses', default=1, help='No. of glimpses to use in the pointer network')
parser.add_argument('--use_tanh', type=str2bool, default=True)
parser.add_argument('--tanh_exploration', default=10, help='Hyperparam controlling exploration in the pointer net by scaling the tanh in the softmax')
parser.add_argument('--dropout', default=0., help='')
parser.add_argument('--terminating_symbol', default='<0>', help='')
parser.add_argument('--beam_size', default=1, help='Beam width for beam search')

# Training
parser.add_argument('--actor_net_lr', default=1e-4, help="Set the learning rate for the actor network")
parser.add_argument('--critic_net_lr', default=1e-4, help="Set the learning rate for the critic network")
parser.add_argument('--actor_lr_decay_step', default=5000, help='')
parser.add_argument('--critic_lr_decay_step', default=5000, help='')
parser.add_argument('--actor_lr_decay_rate', default=0.96, help='')
parser.add_argument('--critic_lr_decay_rate', default=0.96, help='')
parser.add_argument('--reward_scale', default=2, type=float,  help='')
parser.add_argument('--is_train', type=str2bool, default=True, help='')
parser.add_argument('--n_epochs', default=10, help='')
parser.add_argument('--random_seed', default=24601, help='')
parser.add_argument('--max_grad_norm', default=2.0, help='Gradient clipping')
parser.add_argument('--use_cuda', type=str2bool, default=True, help='')
parser.add_argument('--critic_beta', type=float, default=0.9, help='Exp mvg average decay')

# Misc
parser.add_argument('--log_step', default=50, help='Log info every log_step steps')
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--run_name', type=str, default='0')
parser.add_argument('--output_dir', type=str, default='outputs')
parser.add_argument('--epoch_start', type=int, default=0, help='Restart at epoch #')
parser.add_argument('--load_path', type=str, default='')
parser.add_argument('--disable_tensorboard', type=str2bool, default=False)
parser.add_argument('--plot_attention', type=str2bool, default=False)
parser.add_argument('--disable_progress_bar', type=str2bool, default=False)

args = vars(parser.parse_args())

# Pretty print the run args
pp.pprint(args)

# Set the random seed
torch.manual_seed(int(args['random_seed']))

# Optionally configure tensorboard
if not args['disable_tensorboard']:
    configure(os.path.join(args['log_dir'], args['task'], args['run_name']))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('runs/{}_PN'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
# Task specific configuration - generate dataset if needed
### Import data from hdf5 file #############################
# H_file = h5py.File('./8_8_2RB_mob_normal.hdf5','r')
# H_r = np.array(H_file.get('H_r_1'))
# H_i = np.array(H_file.get('H_i_1'))
# H = np.array(H_r + 1j*H_i)
# # print("H shape is:", H.shape)

# H = np.transpose(np.tile(np.transpose(H,(2,1,0)),80),(2,1,0))
# print("H shape is:", H.shape)

# se_max_ur = np.array(H_file.get('se_max_1'))
# # print("se_max_ur shape is:", se_max_ur.shape)
# se_max_ur = np.transpose(np.tile(np.transpose(se_max_ur,(1,0)),80),(1,0))
# print("se_max_ur shape is:", se_max_ur.shape)

# normal_ur = np.array(H_file.get('normal_1'))
# normal_ur = np.tile(normal_ur,80)
# print("normal_ur shape is:", normal_ur.shape)




H_file = h5py.File('./dataset/6416_channel.hdf5','r')
H_r = np.array(H_file.get('H_r'))
H_i = np.array(H_file.get('H_i'))
H_i = np.transpose(H_i,(2,1,0))
H_r = np.transpose(H_r,(2,1,0))
print("H_r shape is:", H_r.shape)
print("H_i shape is:", H_i.shape)
H = np.array((H_r + 1j*H_i)[0:400,:,:])
H = np.transpose(np.tile(np.transpose(H,(2,1,0)),80),(2,1,0))
print("H shape is:", H.shape)


# Import se_max

se_max_ur_file  = h5py.File('./dataset/6416_se_max.hdf5', 'r')
se_max_ur = se_max_ur_file.get('se_max')
se_max_ur = np.array(se_max_ur[0:400,:])
se_max_ur = np.transpose(np.tile(np.transpose(se_max_ur,(1,0)),80),(1,0))
print("se_max_ur shape is:", se_max_ur.shape)


N_file = h5py.File('./dataset/6416_normal.hdf5','r')
normal_ur = np.array(N_file.get('normal'))
normal_ur = np.tile(normal_ur,80)
print("normal_ur shape is:", normal_ur.shape)



# H_file = h5py.File('./dataset/16_4_16_random_all.hdf5','r')
# H_r = np.array(H_file.get('H_r'))
# H_i = np.array(H_file.get('H_i'))
# # H_i = np.transpose(H_i,(2,1,0))
# # H_r = np.transpose(H_r,(2,1,0))
# print("H_r shape is:", H_r.shape)
# print("H_i shape is:", H_i.shape)
# H = np.array(H_r + 1j*H_i)
# H = np.transpose(np.tile(np.transpose(H,(2,1,0)),80),(2,1,0))
# print("H shape is:", H.shape)

# se_max_ur = H_file.get('se_max')
# se_max_ur = np.array(se_max_ur)
# se_max_ur = np.transpose(np.tile(np.transpose(se_max_ur,(1,0)),80),(1,0))
# print("se_max_ur shape is:", se_max_ur.shape)

# N_file = h5py.File('./dataset/16_4_16_random_normal.hdf5','r')
# normal_ur = np.array(N_file.get('normal'))
# normal_ur = np.tile(normal_ur,80)
# print("normal_ur shape is:", normal_ur.shape)

user_set = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
a = 1
b = 1
# size = 4

reward_scale = 1

task = args['task'].split('_')
COP = task[0]
size = int(task[2])
num_ue = int(task[2])
num_bs = int(task[1])
sel_ue = 4
data_dir = 'data/' + COP

train_batch = 250

input_dim = 1

# if COP == 'sort':
#     import sorting_task
    
#     input_dim = 1
#     reward_fn = sorting_task.reward
#     train_fname, val_fname = sorting_task.create_dataset(
#         int(args['train_size']),
#         int(args['val_size']),
#         data_dir,
#         data_len=size)
#     training_dataset = sorting_task.SortingDataset(train_fname)
#     val_dataset = sorting_task.SortingDataset(val_fname)
# elif COP == 'tsp':
#     import tsp_task

#     input_dim = 2
#     reward_fn = tsp_task.reward
#     val_fname = tsp_task.create_dataset(
#         problem_size=str(size),
#         data_dir=data_dir)
#     training_dataset = tsp_task.TSPDataset(train=True, size=size,
#          num_samples=int(args['train_size']))
#     val_dataset = tsp_task.TSPDataset(train=True, size=size,
#             num_samples=int(args['val_size']))
# else:
#     print('Currently unsupported task!')
#     exit(1)

# Load the model parameters from a saved state
if args['load_path'] != '':
    print('  [*] Loading model from {}'.format(args['load_path']))

    model = torch.load(
        os.path.join(
            os.getcwd(),
            args['load_path']
        ))
    model.actor_net.decoder.max_length = size
    model.is_train = args['is_train']
else:
    # Instantiate the Neural Combinatorial Opt with RL module
    model = NeuralCombOptRL(
        input_dim,
        int(args['embedding_dim']),
        int(args['hidden_dim']),
        size, # decoder len
        args['terminating_symbol'],
        int(args['n_glimpses']),
        int(args['n_process_blocks']), 
        float(args['tanh_exploration']),
        args['use_tanh'],
        int(args['beam_size']),
        # reward_fn,
        args['is_train'],
        args['use_cuda'])


save_dir = os.path.join(os.getcwd(),
           args['output_dir'],
           args['task'],
           args['run_name'])    

try:
    os.makedirs(save_dir)
except:
    pass

#critic_mse = torch.nn.MSELoss()
#critic_optim = optim.Adam(model.critic_net.parameters(), lr=float(args['critic_net_lr']))
actor_optim = optim.Adam(model.actor_net.parameters(), lr=float(args['actor_net_lr']))

actor_scheduler = lr_scheduler.MultiStepLR(actor_optim,
        range(int(args['actor_lr_decay_step']), int(args['actor_lr_decay_step']) * 1000,
            int(args['actor_lr_decay_step'])), gamma=float(args['actor_lr_decay_rate']))

#critic_scheduler = lr_scheduler.MultiStepLR(critic_optim,
#        range(int(args['critic_lr_decay_step']), int(args['critic_lr_decay_step']) * 1000,
#            int(args['critic_lr_decay_step'])), gamma=float(args['critic_lr_decay_rate']))

# training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
#     shuffle=True, num_workers=4)

# validation_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)

critic_exp_mvg_avg = torch.zeros(1)
beta = args['critic_beta']

if args['use_cuda']:
    model = model.cuda()
    #critic_mse = critic_mse.cuda()
    critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()

step = 0
val_step = 0

if not args['is_train']:
    args['n_epochs'] = '1'

epoch = int(args['epoch_start'])
history_record = np.zeros((400,num_ue))
reward_record = np.zeros((epoch + int(args['n_epochs']),))
for ep_num in range(epoch, epoch + int(args['n_epochs'])):
    
    if args['is_train']:
        # put in train mode!
        model.train()
        ue_history = 0.01*np.ones((num_ue,))
        # sample = np.zeros((batch_size,input_dim,num_ue))
        r_total = 0
        # sample_batch is [batch_size x input_dim x sourceL]
        for batch_id in range (0,train_batch):
            start_time = time.time()
            # group_idx = usr_group(np.squeeze(H[0,:,:]))
            R = np.zeros((int(args['batch_size']),))
            sample = np.array(np.reshape(se_max_ur[batch_id*int(args['batch_size']):(batch_id+1)*int(args['batch_size']),:],(int(args['batch_size']),input_dim,num_ue)))
            sample = torch.FloatTensor(sample)
            bat = Variable(sample)

            if args['use_cuda']:
                bat = bat.cuda()

            probs, actions, action_idxs = model(bat)
            probs_arr = torch.stack(probs).detach().cpu().numpy()
            actions = torch.stack(actions).cpu().numpy()
            action_idxs = torch.stack(action_idxs).cpu().numpy()
            # print("action_idxs:", action_idxs[:,5])
            # print("Shape:",probs_arr.shape,actions.shape,action_idxs.shape)
            # probs[:,iter_num%int(args['batch_size'])] = np.array(prob)
            # actions[:,iter_num%int(args['batch_size']),:] = np.array(action)
            # action_idxs[:,iter_num%int(args['batch_size'])] = np.array(actions_idx)

            # Reward Function
            mid_time = time.time()
            # print("Inference time is:",mid_time-start_time)
            mod_select = np.ones((sel_ue,)) *16
            for iter_num in range (0,int(args['batch_size'])):
                ue_select = list(np.squeeze(action_idxs[0:sel_ue,iter_num]))
                ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(H[iter_num+batch_id*int(args['batch_size']),:,ue_select],(num_bs,-1)),sel_ue,mod_select)
                # ur_se_total = ur_se_total/normal_ur[iter_num+batch_id*int(args['batch_size'])]

                for idx in range(0,sel_ue):
                    ue_history[ue_select[idx]] += ur_se[idx]
                if (ep_num == epoch + int(args['n_epochs'])-1) and (batch_id <= 2):
                    history_record[batch_id*128+iter_num,:] = ue_history
                if (ep_num == epoch + int(args['n_epochs'])-1) and (batch_id == 3) and (iter_num <= 15):
                    history_record[3*128+iter_num,:] = ue_history

                jfi = np.square((np.sum(ue_history))) / (num_ue * np.sum(np.square(ue_history)))
                R[iter_num] = (a*ur_se_total)*reward_scale

            R = torch.from_numpy(R).float().to(device)
            # R = torch.from_numpy(R).float()
            cal_time = time.time()
            
        
            if batch_id == 0:
                critic_exp_mvg_avg = R.mean()
            else:
                critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())

            advantage = R - critic_exp_mvg_avg
            
            logprobs = 0
            nll = 0
            for prob in probs: 
                # compute the sum of the log probs
                # for each tour in the batch
                logprob = torch.log(prob)
                nll += -logprob
                logprobs += logprob
           
            # guard against nan
            nll[(nll != nll).detach()] = 0.
            # clamp any -inf's to 0 to throw away this tour
            logprobs[(logprobs < -1000).detach()] = 0.

            # multiply each time step by the advanrate
            reinforce = advantage * logprobs
            actor_loss = reinforce.mean()
            
            actor_optim.zero_grad()
           
            actor_loss.backward()

            # clip gradient norms
            torch.nn.utils.clip_grad_norm_(model.actor_net.parameters(),
                    float(args['max_grad_norm']), norm_type=2)

            actor_optim.step()
            actor_scheduler.step()

            critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

            #critic_scheduler.step()

            #R = R.detach()
            #critic_loss = critic_mse(v.squeeze(1), R)
            #critic_optim.zero_grad()
            #critic_loss.backward()
            
            #torch.nn.utils.clip_grad_norm(model.critic_net.parameters(),
            #        float(args['max_grad_norm']), norm_type=2)

            #critic_optim.step()
            
            
            
            writer.add_scalar('avg_reward', R.mean().data, step)
            writer.add_scalar('actor_loss', actor_loss.data, step)
            #writer.add_scalar('critic_loss', critic_loss.data[0], step)
            writer.add_scalar('critic_exp_mvg_avg', critic_exp_mvg_avg.data, step)
            writer.add_scalar('nll', nll.mean().data, step)

            step += 1
            r_total = np.sum(R.cpu().numpy())
            end_time = time.time()
            # print("update time is:", end_time - cal_time)
            # print("total time is:", (end_time-cal_time)+(mid_time-start_time))

    reward_record[ep_num] = r_total
    writer.add_scalar('reward', r_total, ep_num)
    print('Episode',ep_num,':',r_total,'\n')
    # Use beam search decoding for validation
    # model.actor_net.decoder.decode_type = "beam_search"
    
    # print('\n~Validating~\n')

    # # example_input = []
    # # example_output = []
    # avg_reward = []

    # # put in test mode!
    # model.eval()
    # ue_history_test = 0.01*np.ones((num_ue,))
    # sample_test = np.zeros((1,input_dim,num_ue))
    
    # for i in range (0,400):
    #     group_idx = usr_group(np.squeeze(H[i,:,:]))
    #     for m in range (0,num_ue):
    #         sample_test[0,:,m] = np.reshape(np.array([se_max_ur[i,m],ue_history_test[m],group_idx[m]]),(input_dim,1))

    #     bat = Variable(sample_test)

    #     if args['use_cuda']:
    #         bat = bat.cuda()

    #     prob, action, actions_idx = model(bat)

    #     # Reward Function
    #     mod_select = np.ones((sel_ue,)) *16
    #     ue_select = actions_idx[0:sel_ue]
    #     ur_se_total, ur_min_snr, ur_se = data_process(np.reshape(H[i,:,ue_select],(num_bs,-1)),sel_ue,mod_select)
    #     ur_se_total = ur_se_total/normal_ur[i]

    #     for j in range(0,sel_ue):
    #         ue_history_test[ue_select[j]] += ur_se[j]
        
    #     jfi = np.square((np.sum(ue_history_test))) / (num_ue * np.sum(np.square(ue_history_test)))
    #     R_test = (a*ur_se_total + b*jfi)*reward_scale

    #     history_record[ep_num,i,:] = ue_history_test

    #     # avg_reward.append(R_test)
    #     val_step += 1.


     
    if args['is_train']:
        model.actor_net.decoder.decode_type = "stochastic"
         
        print('Saving model...')
     
        torch.save(model, os.path.join(save_dir, '64_mobile_epoch-{}.pt'.format(ep_num)))

        # If the task requires generating new data after each epoch, do that here!
        # if COP == 'tsp':
        #     training_dataset = tsp_task.TSPDataset(train=True, size=size,
        #         num_samples=int(args['train_size']))
        #     training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
        #         shuffle=True, num_workers=1)
        # if COP == 'sort':
        #     train_fname, _ = sorting_task.create_dataset(
        #         int(args['train_size']),
        #         int(args['val_size']),
        #         data_dir,
        #         data_len=size)
        #     training_dataset = sorting_task.SortingDataset(train_fname)
        #     training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
        #             shuffle=True, num_workers=1)
with h5py.File("./result/64_4_16_PN_mobile.hdf5", "w") as data_file:
    data_file.create_dataset("reward", data=reward_record)
    data_file.create_dataset("history", data=history_record)
    # data_file.create_dataset("max_history", data=max_history_record)

print('Training is finished\n')
