import numpy as np
import random
import pybullet as p
from PhysicsEngine import EnvPhysicsEngine


class Drone1DEnv():


    def __init__(self, drone, marker, phi_des):

        self.drone = drone
        self.marker = marker
        self.phi_high = np.deg2rad(90)
        self.phi_dot_high = np.deg2rad(200) # 300
        self.phi_des = phi_des
        self.phi_dot_des = 0.0
        self.action_high = 0.05 #(N-m)
        self.pe = EnvPhysicsEngine()


    # current state--> current position, current velocity in z-direction
    def state(self):
        phi = self.pe.get_currentState()[2]
        phi_dot = self.pe.get_currentState()[5]

        # error representation of state
        state = self.abs_to_error_state(phi, phi_dot)
        return state
    
    # reward
    def reward(self, action):

        phi, phi_dot = self.state()
        t = self.pe.get_time()
        
        if self.done():
            if (abs(phi)>=1.0 or abs(phi_dot)>=1.0 or t>10):
                reward = -10.0
                print('...outbound conditions...', phi, phi_dot, t)
            else:
                reward = 100.0
                print('----desired point achieved----')
        else:            
            reward = -(10*abs(phi) + 0.5*abs(phi_dot) + 0.3*abs(action)) # double them

        return float(reward)
        
    
    # whether goal is achieved or not.
    def done(self):
        t = self.pe.get_time()
        e_phi, e_phi_dot = self.state()
        if (abs(e_phi)>=1.0 or abs(e_phi_dot)>=1.0 or t>=10) or\
            (abs(e_phi)<=np.deg2rad(1)/self.phi_high and\
                abs(e_phi_dot)<=np.deg2rad(0.5)/self.phi_dot_high):
            return True
        return False


    # step
    def step(self, action, phi_des=0.0, phi_dot_des=0):
        # action must be a float.
        action_ = action*self.action_high
        
        self.phi_des = phi_des
        self.phi_dot_des = phi_dot_des
        '''for visualization purpose'''
        state = self.state()
        p.resetBasePositionAndOrientation(self.drone, [0,0,2.0],
                                          p.getQuaternionFromEuler([0,state[0],0]))
        p.resetBasePositionAndOrientation(self.marker, [0,0,2.0],
                                          p.getQuaternionFromEuler([0,np.pi/2+self.phi_des,0]))
        p.stepSimulation()

        '''execution'''
        self.pe.stepSimulation(F=1.236*9.81, M=action_)

        state = self.state()
        reward = self.reward(action)
        done = self.done()
        info = self.info()
        return state, reward, done, info

    # info
    def info(self):
        return {}


    # reset the environment
    def reset(self, obser=None):
        # initializing quadcopter at random angle phi with angular vel phi_dot
        phi, phi_dot = self.random_state_generator(obser)
        p.resetBasePositionAndOrientation(self.drone, [0,0,2.0],
                                          p.getQuaternionFromEuler([0,phi,0]))

        self.pe.reset(phi, phi_dot)

        # return state
        state  = self.abs_to_error_state(phi, phi_dot)
        print('state_reset:', [state[0]*self.phi_high, state[1]*self.phi_dot_high])
        return state


    def random_state_generator(self, obser):

        if obser is None:
            # initialize drone's orientation between -60 to 60 degree.
            phi_init = random.uniform(-np.pi/3, np.pi/3)
            # initialized with angular velocity in between -30 and 30 degree/s.
            phi_dot_init = random.uniform(-np.pi/6, np.pi/6)
             
        else:
            phi_init = obser[0]
            phi_dot_init = obser[1]

        return phi_init, phi_dot_init


    def abs_to_error_state(self, phi, phi_dot):
        # assuming maximum quadcopter angle would be 90 degree
        e_phi = (phi - self.phi_des)/self.phi_high
        e_phi_dot = (phi_dot - self.phi_dot_des)/self.phi_dot_high

        return np.array([e_phi, e_phi_dot])

