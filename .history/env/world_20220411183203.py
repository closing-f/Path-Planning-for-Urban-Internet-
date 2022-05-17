import numpy as np
from env.Cargo import Cargo
import env.utils as U
dynamics = ['point', 'unicycle', 'box2d', 'direct', 'unicycle_acc']
class World(object):
    def __init__(self, world_size, torus, agent_dynamic):
        self.n_agents = None
        # world is square
        self.world_size = world_size
        # dynamics of agents
        assert agent_dynamic in dynamics
        
        self.agent_dynamic = agent_dynamic
        # periodic or closed world
        self.torus = torus
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.cargos=[]
        
        # matrix containing agent states
        self.agent_states = None
        # matrix containing landmark states
        
        # x,y of everything
        self.nodes = None
        self.distance_matrix = None
        self.angle_matrix = None
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.01
        self.action_repeat = 10
        self.timestep = 0
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        self.k_energy=0.5
        self.charge_station=np.zeros(5,2)

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.cargos]

    def reset(self):
        self.timestep = 0
        self.n_agents = len(self.policy_agents)
        pursuers = np.zeros((self.n_agents, 2))
        self.agent_states=pursuers
        
        for i in range(5):
            self.charge_station[i][0]=np.random.randint(0, 20)*5
            self.charge_station[i][1]=np.random.randint(0, 20)*5
        
        for i,agent in enumerate(self.policy_agents):
            agent.reset()
            
        for i,agent in  enumerate(self.scripted_agents):
            agent.reset()
    def step(self):

        self.timestep += 1

        
            
        done=np.zeros(self.n_agents)
   
        for i, agent in enumerate(self.policy_agents):
            cargo_index = int(agent.action.item())
            print(cargo_index)
            # weight = self.cargos[cargo_index.item()].weight
            
           
            
            distance=(agent.position[0]-self.cargos[cargo_index].end_pos[0])**2+(agent.position[1]-self.cargos[cargo_index].end_pos[1])**2
            distance=np.sqrt(distance)
            
            print(agent.energy)
            if(agent.energy==0):
                done[i]=1
            else:
                agent.energy=agent.energy-1
                
                # 开始消耗能量~
                    # 在我们算能量的时候，动作应当从1变成在0.5\\
                agent.position[0]=self.cargos[cargo_index].end_pos[0]
                agent.position[1]=self.cargos[cargo_index].end_pos[1]
                
                agent.cargo_buffer.append(self.cargos[i].wait_time)
                self.cargos[i].take_away=True
                self.cargos[i].wait_time=0

            # uav step
            

        
        return done