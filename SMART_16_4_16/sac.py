import os
import torch
import torch.nn.functional as F
import action_space
import numpy as np
import time
import torch.nn as nn
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy


class SAC(object):
    def __init__(self, num_inputs, num_actions, max_actions, args, lr, alpha_lr):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.action_space = action_space.Discrete_space(max_actions)
        self.k_nearest_neighbors = 1

        self.gpu_ids = [0] if args.cuda and args.gpu_nums > 1 else [-1]

        self.num_actions = num_actions
        self.num_inputs = num_inputs
        self.max_actions = max_actions

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda:0" if args.cuda else "cpu")

        if len(self.gpu_ids) == 1:
            self.critic = QNetwork(num_inputs, num_actions, args.hidden_size).to(self.device)
        if len(self.gpu_ids) > 1:
            self.critic = QNetwork(num_inputs, num_actions, args.hidden_size)
            self.critic = nn.DataParallel(self.critic, device_ids=self.gpu_ids).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        if len(self.gpu_ids) == 1:
            self.critic_target = QNetwork(num_inputs, num_actions, args.hidden_size).to(self.device)
        if len(self.gpu_ids) > 1:
            self.critic_target = QNetwork(num_inputs, num_actions, args.hidden_size)
            self.critic_target = nn.DataParallel(self.critic_target, device_ids=self.gpu_ids).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(num_actions).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=alpha_lr)

            if len(self.gpu_ids) == 1:
                self.policy = GaussianPolicy(num_inputs, num_actions, args.hidden_size).to(self.device)
            if len(self.gpu_ids) > 1:
                self.policy = GaussianPolicy(num_inputs, num_actions, args.hidden_size)
                self.policy = nn.DataParallel(self.policy, device_ids=self.gpu_ids).to(self.device)
            
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            if len(self.gpu_ids) == 1:
                self.policy = DeterministicPolicy(num_inputs, num_actions, args.hidden_size).to(self.device)
            if len(self.gpu_ids) > 1:
                self.policy = DeterministicPolicy(num_inputs, num_actions, args.hidden_size)
                self.policy = nn.DataParallel(self.policy, device_ids=self.gpu_ids).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

    
    def knn_action(self, s_t, proto_action):
        # get the proto_action's k nearest neighbors
        raw_actions, actions = self.action_space.search_point(proto_action, self.k_nearest_neighbors)

        if not isinstance(s_t, np.ndarray):
           s_t = s_t.detach().cpu().numpy()
        # make all the state, action pairs for the critic
        s_t = np.tile(s_t, [raw_actions.shape[1], 1])

        s_t = s_t.reshape(len(raw_actions), raw_actions.shape[1], s_t.shape[1]) if self.k_nearest_neighbors > 1 \
            else s_t.reshape(raw_actions.shape[0], s_t.shape[1])
        raw_actions = torch.FloatTensor(raw_actions).to(self.device)
        s_t = torch.FloatTensor(s_t).to(self.device)

        # evaluate each pair through the critic
        actions_evaluation,_ = self.critic(s_t, raw_actions)
        # print('actions_evaluation:',actions_evaluation,type(actions_evaluation))
        # find the index of the pair with the maximum value
        max_index = np.argmax(actions_evaluation.detach().cpu().numpy(), axis=1)
        max_index = max_index.reshape(len(max_index),)

        raw_actions = raw_actions.detach().cpu().numpy()
        # return the best action, i.e., wolpertinger action from the full wolpertinger policy
        if self.k_nearest_neighbors > 1:
            return raw_actions[[i for i in range(len(raw_actions))], max_index, [0]].reshape(len(raw_actions),1), \
                   actions[[i for i in range(len(actions))], max_index, [0]].reshape(len(actions),1)
        else:
            return raw_actions[max_index], actions[max_index]



    def random_action(self):
        proto_action = np.random.uniform(-1., 1., self.num_actions)
        raw_action, action = self.action_space.search_point(proto_action, 1)
        # print('raw_action and action are:',raw_action,action)
        raw_action = raw_action[0]
        action = action[0]
        assert isinstance(raw_action, np.ndarray)
        # print('raw_action and action is', raw_action,action[0])
        return raw_action, action[0]

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        # print('state shape:', state.shape,'!!!')
        sel_time = time.time()
        if evaluate is False:
            if len(self.gpu_ids) == 1:
                proto_action, _, _ = self.policy.sample(state)
            if len(self.gpu_ids) > 1:
                proto_action, _, _ = self.policy.module.sample(state)
        else:
            if len(self.gpu_ids) == 1:
                _, _, proto_action = self.policy.sample(state)
            if len(self.gpu_ids) > 1:
                _, _, proto_action = self.policy.module.sample(state)
        proto_time = time.time()
        print("proto time is:", proto_time - sel_time)
        
        proto_action = proto_action.detach().cpu().numpy()[0].astype('float64')
        # print('sel proto action is:', proto_action,proto_action.shape)

        raw_action, action = self.knn_action(state, proto_action)
        # print('knn back shape:',raw_action.shape,action.shape)
        knn_time = time.time()
        print("proto time is:", knn_time - proto_time)
        assert isinstance(raw_action, np.ndarray)
        action = action[0]
        raw_action = raw_action[0]
        # print('raw_action and action is', raw_action,action[0])
        return raw_action, action[0]

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        # print("state_batch size:",state_batch.shape)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        # print("next_state_batch size:",next_state_batch.shape)
        # print(next_state_batch.shape,state_batch.shape,action_batch.dtype)
        action_batch = np.reshape(action_batch,(batch_size,1,-1))
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        # print("action_batch size:",action_batch.shape)
        reward_batch = np.reshape(reward_batch,(batch_size,1,-1))
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = np.reshape(mask_batch,(batch_size,1,-1))
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)
        
        # print("reward_batch size:",reward_batch.shape)
        # print("mask_batch size:",mask_batch.shape)

        with torch.no_grad():
            if len(self.gpu_ids) == 1:
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            if len(self.gpu_ids) > 1:
                next_state_action, next_state_log_pi, _ = self.policy.module.sample(next_state_batch)
            # print("next_state_log_pi, next_state_action size:",next_state_log_pi.shape,next_state_action.shape)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            # print("qf1_next_target size:",qf1_next_target.shape)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            # print("min_qf_next_target size:",min_qf_next_target.shape)
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            # print("next_q_value size:",next_q_value.shape)
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        # print("qf1 size:",qf1.shape)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        if len(self.gpu_ids) == 1:
            pi, log_pi, _ = self.policy.sample(state_batch)
        if len(self.gpu_ids) > 1:
            pi, log_pi, _ = self.policy.module.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic.eval()
                self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

