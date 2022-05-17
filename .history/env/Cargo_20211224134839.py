import numpy as np
import env.utils as U
from env.base import Agent


class Cargo(Agent):
    def __init__(self, experiment):
        super(Cargo, self).__init__()
        
        self.weight=np.random.randint(0, 10)
        self.wait_time = 0
        self.take_away=False
        self.end_pos=np.zeros(2)

    def reset(self, ):
        
        end_pos_x = np.random.randint(0, 20)*5
        end_pos_y = np.random.randint(0, 20)*5
        
        self.end_pos[0]= end_pos_x
        self.end_pos[1] = end_pos_y
        
        self.wait_time=np.random.randint(1,10)
        
        self.take_away=False
        
        self.weight=np.random.randint(0, 10)
        
    
    def step(self, world):
        
        self.wait_time+=1

        # nodes = np.vstack([world.agent_states[:, :],
        #                    self.state.p_pos,
        #                    ])
        # distances = U.get_distance_matrix(nodes)
        # evader_dist = distances[-1,:-1]
        
        # self.sub_list = list(np.where(evader_dist < 0.1,True,False))

        
        
        # for i in range(len(self.sub_list)):
        #     if self.sub_list[i] and world.policy_agents[i].energy > 0 and self.take_away==False:
        #        self.take_away=True
        #        world.policy_agents[i].load-=self.weight
        #        world.policy
               
