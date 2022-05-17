import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from itertools import zip_longest
from typing import Dict, List, Tuple, Type, Union, Any
from itertools import chain

LOG_SIG_MAX = 0.05
LOG_SIG_MIN = 0.025
epsilon = 1e-6


class AttentionNet(nn.Module):

    def __init__(self, config, action_space=None, norm_in=True, attend_heads=2):
        super(AttentionNet, self).__init__()
               
        # self.max_nb_UAVs = config.max_nb_UAVs  # 8
        self.time_dim=config.time_dim
        self.n_agents = config.nb_UAVs  # 6
        self.n_cargos=config.nb_cargos
        self.uav_obs_dim = config.uav_obs_dim  # 4
        self.neibor_dim=config.uav_obs_dim
        self.local_obs=config.local_obs
        self.single_obs_all_dim=config.uav_obs_dim
        
        self.me_dim_single = (self.n_agents - 1) * self.uav_obs_dim  # 邻居最大维度 x
        self.uav_state_dim = self.uav_obs_dim + self.local_obs  # 508
        self.features_dim_all = self.n_agents*self.uav_obs_dim+self.n_cargos*config.cargo_dim
        
        self.poi_dim=config.cargo_dim
        self.attend_heads = attend_heads
        self.poi_hidden_dim=config.hidden_dim
        self.neibor_hidden_dim=config.hidden_dim
        self.attend_neibor_dim=self.neibor_hidden_dim // attend_heads # // 整
        self.attend_poi_dim = self.poi_hidden_dim // attend_heads # // 整
        self.action_dim=1
        
        self.state_encoder = nn.Sequential(
            nn.Linear(self.single_obs_all_dim, self.poi_hidden_dim),
            nn.LeakyReLU(),
        )
        
        self.device=config.device
       

        # attention init
        
        # if norm_in:
        #     state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(self.uav_state_dim, affine=False))
        
        
        # * attention_poi
        
        self.key_poi = nn.ModuleList()
        self.selector_poi = nn.ModuleList()
        self.value_poi = nn.ModuleList()
        for i in range(attend_heads):
            self.key_poi.append(nn.Linear(self.poi_dim, self.attend_poi_dim, bias=False))
            self.selector_poi.append(nn.Linear(self.poi_hidden_dim, self.attend_poi_dim, bias=False))
            self.value_poi.append(nn.Sequential(nn.Linear(self.poi_dim, self.attend_poi_dim), nn.LeakyReLU()))
        
        
        # * attention_neibor
        
        self.key_neibor = nn.ModuleList()
        self.selector_neibor = nn.ModuleList()
        self.value_neibor = nn.ModuleList()
        for i in range(attend_heads):
            self.key_neibor.append(nn.Linear(self.neibor_dim, self.attend_neibor_dim, bias=False))
            self.selector_neibor.append(nn.Linear(self.poi_hidden_dim, self.attend_neibor_dim, bias=False))
            self.value_neibor.append(nn.Sequential(nn.Linear(self.neibor_dim, self.attend_neibor_dim), nn.LeakyReLU()))
        
        self.shared_modules = [self.key_poi, self.selector_poi,
                               self.selector_poi, self.key_neibor,self.selector_neibor
                               , self.value_neibor]

    def forward(self, observations):
        
        # input dim is [nr_agent, sum_nei_obs(max_nr, uav_obs)+2+uav_obs] 550
        print(observations.shape)
        observations = torch.reshape(observations, (self.n_agents, -1, self.features_dim_all)) # nb_uav * obs_dim + localobs
        neibor_input=observations[:,:,:(self.n_agents-1)*self.uav_obs_dim]
        
        cargo_input=observations[:,:,self.n_agents*self.uav_obs_dim:]
        
        # print(uav_input.shape)[6, 1, 24]
        # print(cargo_input.shape)[6, 1, 60]
        
        batch_size = observations.shape[1]
        
        
        uav_self_input = observations[:, :, (self.n_agents-1)*self.uav_obs_dim:(self.n_agents)*self.uav_obs_dim] # 6,batchsize,5+2
        
        uav_poi_input= cargo_input[0]
        if_cargos=torch.zeros((self.n_cargos,1 )).to(device=self.device)
        for i in range(self.n_cargos):
            if_cargos[i]=uav_poi_input[0][self.poi_dim*i-1]
            # if(uav_poi_input[i][])
        # print(if_cargos)
        # print(uav_poi_input) 
        # uav_poi_input= observations[0, :, self.me_dim_single+self.uav_obs_dim+self.local_obs:]# 6,batchsize,400
        uav_poi_input=torch.reshape(uav_poi_input,(batch_size, self.n_cargos,-1)).permute(1,0,2)
        
        # print(uav_self_input.shape) 

        state_encoding=torch.zeros((self.n_agents, batch_size, self.poi_hidden_dim)).to(device=self.device)
        
        # print("PoI..........")
        # print(poi_all_values_1)
        
        for i in range(self.n_agents):
            state_encoding[i]=self.state_encoder(uav_self_input[i])
        
        neibor_data = torch.reshape(neibor_input, (self.n_agents, batch_size, self.n_agents-1, self.uav_obs_dim)).permute(0,2,1,3)
        # print(neibor_input.shape)
        # print("PoI..........")
        # print(poi_all_values)
        
        all_head_neibor_keys = [[[ k_ext(k) for k in enc ] for enc in neibor_data ] for k_ext in self.key_neibor]
        # extract sa values for each head for each agent
        all_head_neibor_values = [[[v_ext(v) for v in enc ] for enc in neibor_data ] for v_ext in self.value_neibor]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_neibor_selectors = [[sel_ext(enc) for  enc in (state_encoding)]
                              for sel_ext in self.selector_neibor]
        
        neibor_all_values =torch.zeros((self.n_agents, batch_size, self.poi_hidden_dim)).to(device=self.device)
        all_attend_logits = [[] for _ in range(self.n_agents)]
        all_attend_probs = [[] for _ in range(self.n_agents)]
        # calculate attention per head
        for head_index,curr_head_keys, curr_head_values, curr_head_selectors in zip(range(self.attend_heads)
                ,all_head_neibor_keys, all_head_neibor_values, all_head_neibor_selectors):
            # iterate over agents
            for i, key_i,value_i, selector_i in zip(range(self.n_agents), curr_head_keys,curr_head_values, curr_head_selectors):
                keys = [k for k in key_i ]
                # print(len(keys))
                # print(type(keys[0]))
                values = [v for v in value_i]
                # calculate attention across agents
                attend_logits = torch.matmul(selector_i.view(selector_i.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                # print("scaled_attend_logits")
                # print(scaled_attend_logits)
                # print(scaled_attend_logits.mean())
                
                scaled_attend_logits=scaled_attend_logits / (scaled_attend_logits.mean() + 1e-9)
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                # print(scaled_attend_logits)
                # print("Attend weight_uav")
                # print(attend_weights)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                neibor_all_values[i,:,self.attend_neibor_dim*head_index:self.attend_neibor_dim*(head_index+1)]=other_values
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        
        
        
        
        
        
        
        
        
        
        
        
        # print(neibor_all_values.shape)
        all_head_poi_keys = [[k_ext(enc) for enc in uav_poi_input] for k_ext in self.key_poi]
        # extract sa values for each head for each agent
        all_head_poi_values = [[v_ext(enc) for enc in uav_poi_input] for v_ext in self.value_poi]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_poi_selectors = [[sel_ext(enc) for enc in neibor_all_values] for sel_ext in self.selector_poi]
        
        # print("selector output:")
        # print(all_head_poi_selectors)
        
        poi_all_values = [[] for _ in range(self.n_agents)]
        poi_all_values=torch.zeros((self.n_agents, batch_size, self.action_dim)).to(device=self.device)
        all_attend_logits = [[] for _ in range(self.n_agents)]
        all_attend_probs = [[] for _ in range(self.n_agents)]
        # calculate attention per head
        for head_index, curr_head_keys, curr_head_values, curr_head_selectors in zip(range(self.attend_heads),
                all_head_poi_keys, all_head_poi_values, all_head_poi_selectors):
            # iterate over agents
            for i, selector in zip(range(self.n_agents), curr_head_selectors):
                keys = [k for k in curr_head_keys]
                values = [v for v in curr_head_values]

                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                scaled_attend_logits=scaled_attend_logits / (scaled_attend_logits.mean() + 1e-9)
                # print("scaled_attend_logits")
                # print(scaled_attend_logits)
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                print("PoI Attend weight")

                print(attend_weights)
                for x in range(self.n_cargos):
                    if(if_cargos[x]==1):
                        attend_weights[:,:,x]=0
                # print(attend_weights)
                cargo_index=torch.argmax(attend_weights)
                
                # print(cargo_index)
                if_cargos[cargo_index]=1
                # other_values = (torch.stack(values).permute(1, 2, 0) *
                #                 attend_weights).sum(dim=2)
                # print(other_values.shape)
                poi_all_values[i,:,:]=cargo_index
            
    
        
        
        
        # print("Neibor..........")
        # print(neibor_all_values)
        
        
        attention_state=poi_all_values
        # print(attention_state.shape)
        return attention_state

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)