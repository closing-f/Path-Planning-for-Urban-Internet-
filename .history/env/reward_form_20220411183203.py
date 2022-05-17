import numpy as np

class RewardForm(object):
    def __init__(self, arglist):
        self.nr_agents = arglist.nb_UAVs
        self.n_cargo = arglist.nb_cargos
        self.wait_max_time=10
        

    def reset(self):
        self.timestep = 0
        
    def TimeReward(self, UAVs):
        
        
        reward = np.zeros((self.nr_agents,))
 
        

        for i in range(self.nr_agents):
            reward[i]=UAVs[i].cargo_buffer[-1]/self.wait_max_time
            
        return reward
    
    def PathPlanReward(self,UAVs):
        
        reward=np.zeros((self.nr_agents,))
        
        for i in range(self.nr_agents):
            
            if UAVs[i].action>0.5:
                reward[i]=UAVs[i].energy-UAVs[i].need_step
            if UAVs[i].actioin<0.5:
                reward[i]=UAVs[i].need_step-UAVs[i].energy
            