import numpy as np
import random
import pybullet as p


class Drone1DEnv():


    def __init__(self, drone, marker, phi_des):

        self.drone = drone
        self.marker = marker
        self.obs_high = 60.0
        self.phi_des = phi_des
        self.phi_dot_des = 0.0
        self.offset = 0.0
        self.weight = 1.5*9.81
        self.inertia_yy = 0.052


    # current state--> current position, current velocity in z-direction
    def state(self):
        _, drone_orientation = p.getBasePositionAndOrientation(self.drone)
        _, drone_ang_vel = p.getBaseVelocity(self.drone)

        phi = p.getEulerFromQuaternion(drone_orientation)[1]
        # discretization and error representation of state
        state = self.abs_to_error_state(phi, drone_ang_vel[1])
        return state
    
    
    # whether goal is achieved or not.
    def done(self):
        e_phi, e_phi_dot = self.state()
        if (abs(e_phi)>=1.5 or abs(e_phi_dot)>=8) or\
            (abs(e_phi)<=0.01 and abs(e_phi_dot)<=0.07):
            return True
        return False


    # step
    def step(self, action, phi_des=0.0, phi_dot_des=0):
        # action must be a float.
        action_ = action*(2*self.inertia_yy)
        p.applyExternalTorque(objectUniqueId=self.drone, linkIndex=-1,
                         torqueObj=[0, action_ , 0], #posObj=[0,0,0], 
                         flags=p.LINK_FRAME)
        # self.phi_des = phi_des
        # self.phi_dot_des = phi_dot_des
        p.resetBasePositionAndOrientation(self.marker, [0,0,2.0],
                                          p.getQuaternionFromEuler([0,1.5708,0]))
        p.stepSimulation()
        state = self.state()
        done = self.done()
        return state, done


    # reset the environment
    def reset(self, obser=None):
        # initializing quadcopter with random z_position and z_velocity
        droneStartPos, droneStartOrn, droneStartLinVel, droneStartAngVel\
             = self.random_state_generator(obser)
        p.resetBasePositionAndOrientation(self.drone, droneStartPos,
                                          droneStartOrn)
        p.resetBaseVelocity(self.drone, droneStartLinVel, 
                            droneStartAngVel)

        # return state
        phi = p.getEulerFromQuaternion(droneStartOrn)
        phi_dot = droneStartAngVel
        phi  = phi[1]
        state  = self.abs_to_error_state(phi, phi_dot[1])
        return state


    def random_state_generator(self, obser):

        if obser is None:
            # initialize drone's orientation between -40 to 40 degree.
            phi_init = random.uniform(-0.7, 0.7)
            
            # initialized with angular velocity in between -5 and 5 degree/s.
            phi_dot_init = random.uniform(-0.09,0.09)  
             
        else:
            phi_init = obser[0]
            phi_dot_init = obser[1]

        StartPos = [0,0,2.0] 
        StartOrn = p.getQuaternionFromEuler([0,phi_init,0])
        StartLinVel = [0,0,0]
        StartAngVel = [0,phi_dot_init,0]
        return StartPos, StartOrn, StartLinVel, StartAngVel


    def abs_to_error_state(self, phi, phi_dot):
        # assuming maximum quadcopter and desired point distance would be 5*1.5 meter
        e_phi = (phi - self.phi_des)*1.5
        e_phi_dot = (phi_dot - self.phi_dot_des)*16

        return np.array([e_phi, e_phi_dot])
