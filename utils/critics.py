import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# def weights_init_(m):
#     for mm in m:
#         if isinstance(mm, nn.Linear):
#             print("sss")
#             torch.nn.init.kaiming_normal_(mm)
#             torch.nn.init.constant_(mm.bias, 0)

class AttentionCritic(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, config):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionCritic, self).__init__()
        
        self.time_dim=config.time_dim
        self.device=config.device
        self.hidden_dim=config.hidden_dim
        self.action_encoding_dim=8
        #
        # self.critic_encoders = nn.ModuleList()
        # #todo critics输出的是observation dim 是用来干嘛的
        self.n_agents=config.nb_UAVs
        
        self.critic1 = nn.Sequential(
            nn.Linear(self.hidden_dim+self.action_encoding_dim+self.time_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(self.hidden_dim+self.action_encoding_dim+self.time_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )
        self.action_encoding=nn.Sequential(
            nn.Linear(2, self.action_encoding_dim),
        )
        # self.critic1.apply(weights_init_)
        # self.critic2.apply(weights_init_)
        # self.action_encoding.apply(weights_init_)
    def forward(self, attention_state,position_encode, action):
        """
        Inputs:
        """
        if isinstance(action, list):
            print()
            action = torch.cat(tuple(action), 0)
        # print(s_encoding_tensor.shape)
        # print(action.shape)
        # print(len(other_all_values[0]))
       
        # print(self.action_encoding)
        action_encoding=self.action_encoding(action)
        # print(action_encoding.device)
        sa_encoding=torch.cat([position_encode,attention_state, action_encoding], dim=2)
        # print(sa_encoding.shape)
        # sa_encoding.requires_grad=True
        batch_size=sa_encoding.shape[1]
        # print(batch_size)
        #todo get q_value
        all_rets_q1 = torch.zeros((self.n_agents, batch_size, 1)).to(device=self.device)
        all_rets_q2 = torch.zeros((self.n_agents, batch_size, 1)).to(device=self.device)
        for i in range(self.n_agents):
            # print(sa_encoding.shape)
            q1 = self.critic1(sa_encoding[i])
            q2 = self.critic2(sa_encoding[i])
            all_rets_q1[i] = q1
            all_rets_q2[i] = q2
        # print(all_rets_q1.shape)
        return all_rets_q1, all_rets_q2