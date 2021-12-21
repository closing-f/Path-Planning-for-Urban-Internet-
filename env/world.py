import numpy as np
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
        
        for i,agent in enumerate(self.policy_agents):
            agent.reset()
            
        for i,agent in enumerate(self.scripted_agents):
            agent.reset()
    def step(self):

        self.timestep += 1

        if self.agent_dynamic == 'unicycle':
            
            scaled_actions = np.zeros([self.n_agents, 2])
            
            for i, agent in enumerate(self.policy_agents):
                scaled_actions[i, 0] = agent.action[0]
                scaled_actions[i, 1] = agent.action[1]

                # 这里来改变能量
                # 当能量小于0的时候，就不允许动了
                # 我这里没有写其他的功能，但是可以在这里实现一下，比如自己飞回来之类的
                # print("energy",end=' ')
                # print(agent.energy)
                
                if agent.energy < 0.00001:
                    scaled_actions[i, 0] = 0
                    scaled_actions[i, 1] = 0
                    agent.energy = 0
                else:
                    # 开始消耗能量~
                        # 在我们算能量的时候，动作应当从1变成在0.5\\
                    v = np.sqrt((agent.action** 2).sum(axis=-1)) * 0.5
                    # print(v)
                    
                    delta_E = 1/3.0 * v ** 3 - 0.0625 * v + 0.03
                    delta_E = delta_E / 15
                    delta_E += 0.001

                    agent.energy = agent.energy - delta_E
                    if agent.energy < 0.00001:
                        agent.energy = 0

            step = np.concatenate([scaled_actions[:, [0]],
                                    scaled_actions[:, [1]]],
                                    axis=1)
            next_coord = self.agent_states[:, 0:2] + step
            
            if self.torus: # cycle 
                next_coord = np.where(next_coord < 0, next_coord + self.world_size, next_coord)
                next_coord = np.where(next_coord > self.world_size, next_coord - self.world_size, next_coord)
            else: # not cycle
                next_coord = np.where(next_coord < 0, 0, next_coord)
                next_coord = np.where(next_coord > self.world_size, self.world_size, next_coord)

            agent_states_next = next_coord

            self.agent_states = agent_states_next
            
            # uav step
            for i, agent in enumerate(self.policy_agents):
                agent.state.p_pos = self.agent_states[i, 0:2]
            
            # cargo step
            for i, agent in enumerate(self.scripted_agents):
                action = agent.step(self)

        
        return 0