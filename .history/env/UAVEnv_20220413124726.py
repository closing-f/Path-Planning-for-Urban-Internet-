import gym
import numpy as np
from gym import spaces
from env.reward_form import RewardForm
from env.world import World
from env.UAV import PointAgent
from env.Cargo import Cargo
# import matplotlib.pyplot as plt
import torch
# from ps_argument import parse_args

class UAVnet(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, arglist):
        self.n_agnent = arglist.nb_UAVs 
        self.n_cargos = arglist.nb_cargos
        # self.comm_radius = arglist.comm_radius
        self.world_size = arglist.world_size
        self.excel_index=1 
        self.max_energy=arglist.max_energy
        self.max_weight=arglist.max_weight
        self.torus = arglist.torus # true->world is cycle, false->world is rectangle
        self.uav_obs_dim = arglist.uav_obs_dim
        self.cargo_dim = arglist.cargo_dim
        self.obs_dim_single_all = (arglist.nb_UAVs * arglist.uav_obs_dim+self.n_cargos*self.cargo_dim)
        self.action_dim = arglist.action_dim
      
        self.n_step = arglist.n_step
        # other parameters
        self.obs_mode = arglist.obs_mode  # default='sum_obs_no_ori'
        self.dynamics = arglist.dynamics #  default='unicycle'
        self.RewardForm = RewardForm(arglist)
        self.world = World(arglist.world_size, arglist.torus, arglist.dynamics)
        self.world_size=arglist.world_size
        self.world.agents = [PointAgent(self) for _ in range(self.n_agnent)]
        self.world.cargos = [Cargo(self) for _ in range(self.n_cargos)]
        # [self.world.agents.append(Cargo(self)) for _ in range(self.n_cargo)]
    @property # 对外的接口
    def policy_agents(self):
        return self.world.policy_agents
    
    @property
    def scriped_agents(self): # 对外的接口
        return self.world.cargos
    
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
        :return obs_reset -> list
        """
        self.RewardForm.reset()
        self.world.agents = [PointAgent(self) for _ in range(self.n_agnent)]
        self.world.cargos = [Cargo(self) for _ in range(self.n_cargos)]
        self.world.reset()
        # print(self.world.agents) 106个
        obs_reset = []
        
        for i, agent in enumerate(self.world.policy_agents):
            obs= agent.get_observation(i,
                                        self.world.cargos,
                                        self.world.agents,
                                     )
            
            obs_reset.append(obs)
            
        return obs_reset

    def step(self, actions=None):
        """
        :param actions: UAV_actions -> self.n_agnent * action_dim = self.n_agnent * 2
        :return: next_obs, r, dones -> 1 is terminal, info -> List[Dict[str, Any]]
        """
        self.RewardForm.timestep = self.RewardForm.timestep+1
        # print("$$actions")
        # print(actions)
        assert len(actions) == self.n_agnent
        # print(actions.shape)
        for agent, action in zip(self.policy_agents, actions):
            action = action.flatten()
            
            agent.action = action
            
        # 距离原点的距离
        dones= self.world.step()
        next_obs = []
        for i, bot in enumerate(self.world.policy_agents):
            ob = bot.get_observation(i,
                                    self.world.cargos,
                                    self.world.agents,
                                   )

            next_obs.append(ob)
        
        r = self.RewardForm.TimeReward(self.world.agents)
        
        

        info = [{'pursuer_states': self.world.agent_states,
                'actions': actions}]
        
        return next_obs, r, dones, info

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
