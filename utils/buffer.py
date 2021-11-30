import numpy as np
from torch import Tensor
from torch.autograd import Variable
import json
class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, config):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        
        self.obs_dim_single_all = (config.nb_UAVs * config.uav_obs_dim) + config.local_obs 
        self.obs_dims=[self.obs_dim_single_all for i in range(config.nb_UAVs)]
        self.ac_dims=[config.dim_a for i in range(config.nb_UAVs)]
        self.share_obs_dim=config.nb_PoIs * config.poi_dim
        self.max_steps = config.buffer_length
        self.num_agents = config.nb_UAVs
        self.gamma=config.gamma
        self.obs_buffs = []
        self.ac_buffs = []
        self.log_ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.share_obs_buffs=[]
        self.next_share_obs_buffs=[]
        self.return_buffs=[]
        # print("$$MAX_STEPS")
        # print(max_steps)
        for odim, adim in zip(self.obs_dims, self.ac_dims):
            self.obs_buffs.append(np.zeros((self.max_steps, odim), dtype=np.float32))
            self.ac_buffs.append(np.zeros((self.max_steps, adim), dtype=np.float32))
            self.log_ac_buffs.append(np.zeros(self.max_steps, dtype=np.float32))
            self.rew_buffs.append(np.zeros(self.max_steps, dtype=np.float32))
            self.return_buffs.append(np.zeros(self.max_steps, dtype=np.float32))
            self.next_obs_buffs.append(np.zeros((self.max_steps, odim), dtype=np.float32))
            self.done_buffs.append(np.zeros(self.max_steps, dtype=np.uint8))

        self.share_obs_buffs.append(np.zeros((self.max_steps, self.share_obs_dim), dtype=np.float32))
        self.next_share_obs_buffs.append(np.zeros((self.max_steps, self.share_obs_dim), dtype=np.float32))
        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, share_obs,observations, actions, log_action, rewards, next_share_obs,next_observations, dones):
        # print("$$buffer dim")
        nentries = 1
        # print(nentries)# handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.log_ac_buffs[agent_i] = np.roll(self.log_ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
            self.share_obs_buffs[0]=np.roll(self.share_obs_buffs[0],
                                                   rollover)
            self.next_share_obs_buffs[0]=np.roll(self.next_share_obs_buffs[0],
                                                   rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = observations[agent_i]
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.log_ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = log_action[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = next_observations[agent_i]
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[agent_i]
        self.share_obs_buffs[0][self.curr_i:self.curr_i + nentries]=share_obs[0]
        self.next_share_obs_buffs[0][self.curr_i:self.curr_i + nentries]=next_share_obs[0]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample(self, N, to_gpu=False, norm_rews=False):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=True)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.share_obs_buffs[0][inds])],
                [cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.log_ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.return_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.next_share_obs_buffs[0][inds])],
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)])
    def compute_returns(self,last_reward):
        # print(last_reward)
        for i in range(self.num_agents):
            self.return_buffs[i][-1]=last_reward[i]
        for i in range(self.num_agents): 
            for step in reversed(range(self.rew_buffs[i].shape[0]-1)):
                    self.return_buffs[i][step] = self.return_buffs[i][step + 1] * self.gamma + self.rew_buffs[i][step]
        
    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]
    def store(self,filename):

        temp = [[],[],[],[],[],[],[],[]]
        for j in range(self.num_agents):
            temp[0].append(self.obs_buffs[j].tolist())
            temp[1].append(self.ac_buffs[j].tolist())
            temp[2].append(self.log_ac_buffs[j].tolist())
            temp[3].append(self.rew_buffs[j].tolist())
            temp[4].append(self.next_obs_buffs[j].tolist())
            temp[5].append(self.done_buffs[j].tolist())
        temp[6].append(self.share_obs_buffs[0].tolist())
        temp[7].append(self.next_share_obs_buffs[0].tolist())
        import os
        add=os.path.exists(filename)
        if add==True:
            with open(filename, 'r+') as file:
                p = json.load(file)
                for m in range(6):
                    for k in range(self.num_agents):
                        for i in range(len(temp[m][k])):
                            p[m][k].append(temp[m][k][i])
                for n in range(6,8):
                    for i in range(len(temp[n][0])):
                        p[n][0].append(temp[n][0][i])
            with open(filename, 'w') as file_obj:
                json.dump(p, file_obj)
        else:
            with open(filename, 'w') as file_obj:
                json.dump(temp, file_obj)
        # print(type(replay_buffer.obs_buffs)) list
        # print(len(replay_buffer.obs_buffs)) 6
        # print(type(replay_buffer.obs_buffs[0])) array
        # print(replay_buffer.ac_buffs)
        # print(total)

    def clear(self):
        self.obs_buffs = []
        self.ac_buffs = []
        self.log_ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.share_obs_buffs=[]
        self.next_share_obs_buffs=[]
        self.return_buffs=[]
        # print("$$MAX_STEPS")
        # print(max_steps)
        for odim, adim in zip(self.obs_dims, self.ac_dims):
            self.obs_buffs.append(np.zeros((self.max_steps, odim), dtype=np.float32))
            self.ac_buffs.append(np.zeros((self.max_steps, adim), dtype=np.float32))
            self.log_ac_buffs.append(np.zeros(self.max_steps, dtype=np.float32))
            self.rew_buffs.append(np.zeros(self.max_steps, dtype=np.float32))
            self.return_buffs.append(np.zeros(self.max_steps, dtype=np.float32))
            self.next_obs_buffs.append(np.zeros((self.max_steps, odim), dtype=np.float32))
            self.done_buffs.append(np.zeros(self.max_steps, dtype=np.uint8))

        self.share_obs_buffs.append(np.zeros((self.max_steps, self.share_obs_dim), dtype=np.float32))
        self.next_share_obs_buffs.append(np.zeros((self.max_steps, self.share_obs_dim), dtype=np.float32))
        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def load(self, filename):

        self.obs_buffs = []
        self.ac_buffs = []
        self.log_ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        self.share_obs_buffs=[]
        self.next_share_obs_buffs=[]
        with open(filename, 'r+') as file:
            p = json.load(file)
            buffer_size=len(p[1][1])
            print(buffer_size)
            for j in range(self.num_agents):
                self.obs_buffs.append(np.array(p[0][j],dtype=np.float32))
                self.ac_buffs.append(np.array(p[1][j], dtype=np.float32))
                self.log_ac_buffs.append(np.array(p[2][j], dtype=np.float32))
                self.rew_buffs.append(np.array(p[3][j], dtype=np.float32))
                self.next_obs_buffs.append(np.array(p[4][j], dtype=np.float32))
                self.done_buffs.append(np.array(p[5][j], dtype=np.float32))
            self.share_obs_buffs.append(np.array(p[6][0], dtype=np.float32))
            self.next_share_obs_buffs.append(np.array(p[7][0], dtype=np.float32))
            if self.max_steps<buffer_size:
                self.max_steps=buffer_size
                self.filled_i=buffer_size
                self.curr_i=0
            else:
                self.curr_i=buffer_size
                self.filled_i=buffer_size

