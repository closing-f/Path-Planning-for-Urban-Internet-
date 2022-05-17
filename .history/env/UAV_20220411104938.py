import numpy as np
import random
from gym import spaces
from env.base import Agent
from itertools import zip_longest

class PointAgent(Agent):
    def __init__(self, experiment):
        super(PointAgent, self).__init__()
        self.nb_cargos=experiment.n_cargos
        self.cargo_dim=experiment.cargo_dim
        self.nb_UAVs=experiment.n_agnent
        self.uav_obs_dim=experiment.uav_obs_dim
        self.max_energy = experiment.max_energy
        self.energy =  self.max_energy
        self.weight=experiment.max_weight
        self.max_weight=experiment.max_weight
        self.world_size=experiment.world_size
        self.max_wait_time=10
        self.position=None
        self.action=None
        self.cargo_buffer=[]
        
        #记录当前该运送哪一个货物
        self.cargo_step=0

    def reset(self,):
        # 还是局限在一个 100的范围内
        
        
        self.energy = self.max_energy
        self.weight=self.max_weight  
        self.delta_energy = 0
        self.cargo_buffer=[]
        
        self.cargo_step=0
        self.weight_self=0
        self.position=np.zeros(2)

    def get_observation(self,uav_index,cargos_info,uav_info):
        
      
        # print(evader_deltax)
        cargo_obs = np.zeros((self.nb_cargos,self.cargo_dim))
        uav_obs = np.zeros((self.nb_UAVs,self.uav_obs_dim))
 
        for i in range(self.nb_cargos):
            
            # 位置信息
            cargo_obs[i, 0] = cargos_info[i].end_pos[0]/self.world_size
            cargo_obs[i, 1] = cargos_info[i].end_pos[1]/self.world_size
            # 等待时间
            cargo_obs[i, 2] = cargos_info[i].wait_time/self.max_wait_time
            
            # 是否被取走，可以放在全局环境中
            cargo_obs[i, 3] = cargos_info[i].take_away
       
        for i in range(self.nb_UAVs):
            if(i==uav_index):
                uav_obs[-1, 0] = uav_info[i].energy/self.max_energy
                
                #载重
                uav_obs[-1, 1] = uav_info[i].weight
                # uav_obs[-1, 1] = 0
                uav_obs[-1, 2] = uav_info[i].position[0]/self.world_size
                uav_obs[-1, 3] = uav_info[i].position[1]/self.world_size
            else:
                uav_obs[num, 0] = uav_info[i].energy/self.max_energy
                #载重,抽象为次数
                uav_obs[num, 1] = uav_info[i].weight
                uav_obs[num, 2] = uav_info[i].position[0]/self.world_size
                uav_obs[num, 3] = uav_info[i].position[1]/self.world_size
                num+=1
                   
        uav_obs=uav_obs.reshape(self.uav_obs_dim*self.nb_UAVs)
          
        cargo_obs=cargo_obs.reshape(self.cargo_dim*self.nb_cargos)
        obs=np.hstack([uav_obs.flatten(), cargo_obs.flatten()])
        return obs


    
    
    