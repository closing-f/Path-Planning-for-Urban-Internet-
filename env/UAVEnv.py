import gym
import numpy as np
from gym import spaces
from env.reward_form import RewardForm
from env.base import World
from env.UAV import PointAgent
from env.Cargo import Cargo
import matplotlib.pyplot as plt
import torch
# from ps_argument import parse_args

class UAVnet(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, arglist):
        self.n_agnent = arglist.nb_UAVs 
        self.n_cargo = arglist.nb_cargos
        # self.comm_radius = arglist.comm_radius
        self.world_size = arglist.world_size
        self.excel_index=1 
    
        self.torus = arglist.torus # true->world is cycle, false->world is rectangle
        self.uav_obs_dim = arglist.uav_obs_dim
        self.cargo_dim = arglist.cargo_dim
        self.obs_dim_single_all = (arglist.nb_UAVs * arglist.uav_obs_dim)
        self.action_dim = arglist.action_dim
        
        self.n_step = arglist.n_step
        # other parameters
        self.obs_mode = arglist.obs_mode # default='sum_obs_no_ori'
        self.dynamics = arglist.dynamics #  default='unicycle'
        self.RewardForm = RewardForm(arglist)
        self.world = World(arglist.world_size, arglist.torus, arglist.dynamics)
        
        self.world.agents = [PointAgent(self) for _ in range(self.n_agnent)]
        self.world.cargos= [Cargo(self) for _ in range(self.n_cargo)]
        # [self.world.agents.append(Cargo(self)) for _ in range(self.n_cargo)]
    @property
    def agents(self):
        return self.world.policy_agents

    @property
    def observation_space(self):
        ob_space = spaces.Box(low=0., high=1., shape=(self.obs_dim_single_all,), dtype=np.float32)
        return ob_space



    @property
    def action_space(self):
        return spaces.Box(low=0., high=+1., shape=(self.action_dim,), dtype=np.float32)

    @property
    def timestep_limit(self):
        return self.n_step

    @property
    def is_terminal(self):
        if self.RewardForm.timestep >= self.timestep_limit:
            return True
        return False

    def reset(self):
        """
        dim_rec_o = (self.max_nr_agents - 1, self.uav_obs_dim)
        dim_evader_o = (self.n_cargo, self.cargo_dim)
        obs_reset = []
        for i in range(self.n_agnent):
            sum_nei_obs = np.ones(dim_rec_o)
            # local and time is + 2
            self_obs = np.zeros(self.uav_obs_dim + 2)
            sum_evader_obs = np.ones(dim_evader_o)
            ob_current_i = np.hstack([sum_nei_obs.flatten(), self_obs.flatten(), sum_evader_obs.flatten()])
            obs_reset.append(ob_current_i)
        # print(obs_reset)
        obs_reset = np.array(obs_reset)
        :return: obs_reset -> list
        """
        self.RewardForm.reset()
        self.world.agents = [PointAgent(self) for _ in range(self.n_agnent)]
        [self.world.agents.append(Cargo(self)) for _ in range(self.n_cargo)]
        pursuers = np.zeros((self.n_agnent, 3))
        evader = np.zeros((100, 2))
        for i in range(10):
            for j in range(10):
                evader[int(i * 10 + j)][0] = 50 + i * 100
                evader[int(i * 10 + j)][1] = 50 + j * 100
        self.world.agent_states = pursuers
        self.world.landmark_states = evader
        self.world.reset()
        # print(self.world.agents) 106个
        obs_reset = []
        share_obs=[]
        aa=True
        for i, bot in enumerate(self.world.policy_agents):
            shareobs,ob= bot.get_observation(
                                     self.world.distance_matrix[i, :],
                                     self.world.delta_matrix[i],
                                     self.world.agents,
                                     self.RewardForm.timestep
                                     )
            obs_reset.append(ob)
            if aa:
                share_obs.append(shareobs)
                aa=False

        # print("obs_reset: {}".format(obs_reset))
        return share_obs, obs_reset

    def step(self, actions=None):
        """
        dim_rec_o = (self.max_nr_agents - 1, self.uav_obs_dim)
        dim_evader_o = (self.n_cargo, self.cargo_dim)
        new_obs = []
        infos = []
        for i in range(self.n_agnent):
            sum_nei_obs = np.ones(dim_rec_o)
            # local and time is + 2
            self_obs = np.zeros(self.uav_obs_dim + 2)
            sum_evader_obs = np.ones(dim_evader_o)
            ob_current_i = np.hstack([sum_nei_obs.flatten(), self_obs.flatten(), sum_evader_obs.flatten()])
            new_obs.append(ob_current_i)
        # print(new_obs)
        rewards = np.zeros(self.n_agnent, dtype=np.float32)
        # print(rewards)
        dones = np.zeros(self.n_agnent)
        new_obs = np.array(new_obs)
        :param actions: UAV_actions -> self.n_agnent * action_dim = self.n_agnent * 2
        :return: next_obs, r, dones -> 1 is terminal, info -> List[Dict[str, Any]]
        """
        self.RewardForm.timestep = self.RewardForm.timestep+1
        # print("$$actions")
        # print(actions)
        assert len(actions) == self.n_agnent
        # print(actions.shape)
        for agent, action in zip(self.agents, actions):
            action = action.flatten()
            
            # print("action_1： {}".format(action[1]))
            
            agent.action = action[0:2]
            
            # print(agent.action)
            # if self.world.dim_c > 0:
            #     agent.action.c = action[2:]
        # 距离原点的距离
        distance_now = self.world.step()
        next_obs = []
        dones = np.zeros(self.n_agnent)
        n_share_obs=[]
        aa=True
        for i, bot in enumerate(self.world.policy_agents):
            # print(hop_counts)
            # bot_in_subset = [list(s) for s in sets if i in s]
            # [bis.remove(i) for bis in bot_in_subset]
            next_share_obs,ob = bot.get_observation(self.world.distance_matrix[i, :],
                                     self.world.delta_matrix[i],
                                     self.world.agents,
                                     self.RewardForm.timestep)
            
            # print(len(ob))
            next_obs.append(ob)
            if aa:
                n_share_obs.append(next_share_obs)
                aa=False
            dones[i] = self.is_terminal
        if self.reward_form == 'LT':
            r = self.RewardForm.LT(self.world.agents)
        if self.reward_form == 'PC':
            r = self.RewardForm.PC(self.world.agents)
        if self.reward_form == 'IA':
            r = self.RewardForm.IA(self.world.agents)
        if self.reward_form == 'RE':
            r = self.RewardForm.RE(self.world.agents)
        if self.reward_form == 'AL':
            r = self.RewardForm.AL(self.world.agents)


        info = [{'pursuer_states': self.world.agent_states,
                'evader_states': self.world.landmark_states,
                'actions': actions,
                'distance': distance_now}]

        # if dones[0] == True:
        #     print("The env is reset")
        #     self.reset()

        return n_share_obs,next_obs, r, dones, info

    def render(self, ws, mode='human'):
        for i in range(self.n_agnent):
            ws.write(self.excel_index, i*2, self.world.agent_states[i, 0])
            ws.write(self.excel_index, i*2+1, self.world.agent_states[i, 1])
        self.excel_index += 1
    
    def close(self):
        pass

if __name__ == "__main__":
    arglist = parse_args()
    env = UAVnet(arglist)
    # env.step()
    print(env.action_space.shape)
    print(env.observation_space)
