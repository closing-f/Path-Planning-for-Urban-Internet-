import numpy as np
import env.utils as U

dynamics = ['point', 'unicycle', 'box2d', 'direct', 'unicycle_acc']


class EntityState(object):
    """physical/external base state of all entities"""
    def __init__(self):
        # physical position
        self.p_pos = None
        self.start_pos=None
        

       


class AgentState(EntityState):
    """state of agents (including communication and internal/mental state)"""
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

class Entity(object):
    """properties and state of physical world entity"""
    def __init__(self):
        # name
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

class Agent(Entity):
    """properties of agent entities"""
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
 
        # script behavior to execute
        self.action_callback = None
        # physical damping
        self.lin_damping = 0.01  # 0.025  # 0.05
        self.ang_damping = 0.01  # 0.05
        self.max_lin_velocity = 50  # cm/s
        self.max_ang_velocity = np.pi  # 2 * np.pi  # rad/s
        self.max_lin_acceleration = 10  # 25  # 100  # cm/s**2
        self.max_ang_acceleration = np.pi  # 60  # rad/s**2