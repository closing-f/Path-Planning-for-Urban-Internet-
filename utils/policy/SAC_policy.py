import numpy as np
import torch 
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union, Any
from itertools import chain
LOG_SIG_MAX = 0.1
LOG_SIG_MIN = 0.05
EPSILON = 1e-6

class SAC(nn.Module):
    # CustomNetwork 直接写固定
    def __init__(self, config, action_space=None, norm_in=True, attend_heads=4):
        super(SAC, self).__init__()

        # build the network
        self.action_scale=1
        
        self.n_agents = config.nb_UAVs
        self.dim_a=config.dim_a
        self.device=config.device
        self.time_dim=config.time_dim
        self.hidden_dim=config.hidden_dim
        self.policy_net = nn.Sequential(
            nn.Linear(self.hidden_dim+self.time_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 4)
        )
        self.time_encoding = nn.Sequential(
            nn.Linear(self.time_dim, 8),
            nn.LeakyReLU(),
        )
        # self.apply(weights_init_)
    def forward(self, attention_state, position_encoding,update_critic=False):

        batch_size = attention_state.shape[1]
        mean_total = torch.zeros((self.n_agents, batch_size, self.dim_a)).to(device=self.device)
        std_total = torch.zeros((self.n_agents, batch_size, self.dim_a)).to(device=self.device)
        actor_input=torch.cat([position_encoding,attention_state], dim=2)
        for i in range(self.n_agents):
            actor_output = self.policy_net(actor_input[i])
            # print(actor_output.shape)
            mean,std = actor_output[:,0:2], actor_output[:,2:]

            mean_total[i, :] = mean
            std = torch.clamp(std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            std_total[i, :] = std
        normal = Normal(mean_total, std_total)

        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t=torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log((self.action_scale - y_t.pow(2)) + EPSILON)
        log_prob = log_prob.sum(2, keepdim=True)
        
        return y_t, log_prob