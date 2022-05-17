from math import exp
import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union, Any
from itertools import chain
LOG_SIG_MAX = 0.05
LOG_SIG_MIN = 0.025
epsilon = 1e-6


class DDPG(nn.Module):
    # CustomNetwork 直接写固定
    def __init__(self, config, action_space=None, norm_in=True, attend_heads=4):
        super(DDPG, self).__init__()
        self.action_scale=1
        self.n_agents = config.nb_UAVs
        self.dim_a=config.action_dim
        self.device=config.device
        self.time_dim=config.time_dim
        # self.time_dim=0
        self.hidden_dim=config.hidden_dim
        self.policy_net = nn.Sequential(
            nn.Linear(self.hidden_dim+self.time_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 4)
        )
        self.time_encoding = nn.Sequential(
            nn.Linear(self.time_dim, 8),
            nn.LeakyReLU(),
        )
        # self.policy_net.apply(weights_init_)
    def forward(self, attention_state, position_encoding):
        # print(s_encoding_tensor.shape)
        batch_size = attention_state.shape[1]
        mean_total = torch.zeros((self.n_agents, batch_size, self.dim_a)).to(device=self.device)
        std_total = torch.zeros((self.n_agents, batch_size, self.dim_a)).to(device=self.device)
        # position=self.time_encoding(position_encoding)
        actor_input=torch.cat([position_encoding,attention_state], dim=2)
        
        for i in range(self.n_agents):
            actor_output = self.policy_net(actor_input[i])
            # print(actor_output.shape)
            mean, std = actor_output[:,0:2], actor_output[:,2:]
            mean_total[i, :] = mean
            std_total[i, :] = std

        std_total = std_total.sum(2, keepdim=True)
        y_t = torch.tanh(mean_total)
        return y_t, std_total