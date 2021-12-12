import numpy as np
import random
import gym
from gym import spaces
from PhysicsEngine import EnvPhysicsEngine
from render import renderEnv

# Drone Environment

'''
get_currentstate = [x_dot, z_dot, theta_dot, theta_p_dot,
                    x, z, theta, theta_p]
'''
class Quad2DPend_env(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self, render_= False):
        super(Quad2DPend_env, self).__init__()

        self.observation_space = spaces.Box(low=-90, high=90,
                                            shape=(2,), 
                                            dtype=np.float32)

        self.action_space = spaces.Box(low = -1, high = 1,
                                       shape = (1,), 
                                       dtype=np.float32)

        self.theta_p_des = 0.0
        self.theta_p_dot_des = 0.0
        self.theta_max = 45
        self.render_ = render_
        self.pe = EnvPhysicsEngine()

        
    # current state--> current position, current velocity in z-direction
    def state(self):

        theta_p_dot = self.pe.get_currentState()[3]
        theta_p = self.pe.get_currentState()[7]

        # error representation of state
        state = self.abs_to_error_state(theta_p, theta_p_dot)

        return state


    # reward
    def reward(self, action):

        theta_p, theta_p_dot = self.state()
        t = self.pe.get_time()
        
        if self.done():
            if (abs(theta_p)>=90 or abs(theta_p_dot)>=90 or t>30):
                reward = -10.0
                print('...outbound conditions...', theta_p, theta_p_dot, t)
            else:
                reward = 100.0
                print('----desired point achieved----')
        else:            
            reward = -(0.4*abs(theta_p) + 0.04*abs(theta_p_dot) + 0.02*abs(action))

        return float(reward)
    
    
    # whether goal is achieved or not.
    def done(self):
        theta_p, theta_p_dot = self.state()
        t = self.pe.get_time()
        '''
        done=1; episodes terminates when:
          1. if theta_p >= 60: if pendulum from verticle axis is at angle 60 or more.
          2. if theta_p_dot >= 5: if pendulum angular velocity is >=5 rad/s {pysical constraints}.
          3. if theta_p<=1 and theta_p_dot<=0.05: desired condition is achieved.
        '''
        if (abs(theta_p)>=90 or abs(theta_p_dot)>=90 or t>30) or\
            (abs(theta_p)<=1 and abs(theta_p_dot)<=0.05):
            return True

        return False


    #info
    def info(self):
        return {}


    # step
    def step(self, action):
        # action must be a float.
        # action is theta angle that drone should achieve.
        action_ = action*self.theta_max

        #from pd controller get F, M.
        F, M = self.pe.get_effort(action_)
        self.pe.stepSimulation(F, M)
    
        state = self.state()
        reward = self.reward(action)
        done = self.done()
        info = self.info()
        if self.render_:
          self.render()

        return state, reward, done, info


    # reset the environment
    def reset(self):
        # initializing pendulum with random angle.
        theta_p_init = self.random_state_generator()
        self.pe.reset(theta_p_init)
        
        # if render = True initialize the animation
        if self.render_:
          self.re = renderEnv()

        # return state
        return self.state()


    def random_state_generator(self):
        # initialize pendulum with random angle from verticle
        theta_p_init = random.uniform(-1.0, 1.0)*30
        
        return theta_p_init


    # absolute to error conversion
    def abs_to_error_state(self, theta_p, theta_p_dot):

        e_theta_p = theta_p - self.theta_p_des
        e_theta_p_dot = theta_p_dot - self.theta_p_dot_des
    
        return np.array([e_theta_p, e_theta_p_dot])

    # rendering environment
    def render(self, mode='human'):
        
        self.re.show(self.pe.get_currentState())
        pass