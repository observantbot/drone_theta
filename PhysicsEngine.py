from numba import jit
import numpy as np

@jit(nopython=True)
def get_sderivative(F, M, S, m, I, g):

    phi = S[2]
    s_dot = np.zeros(6)
    
    s_dot[:3] = S[3:]

    s_dot[3] = F/m * np.sin(phi)

    s_dot[4] = F/m * np.cos(phi) - g

    s_dot[5] = M/I

    return s_dot


@jit(nopython = True)
def update(F, M, curr, t, delta_t, m, I, g):

    t += delta_t

    k1 = delta_t*get_sderivative(F, M, curr, m, I, g)

    k2 = delta_t*get_sderivative(F, M, curr + k1/2, m, I, g)

    k3 = delta_t*get_sderivative(F, M, curr + k2/2, m, I, g)

    k4 = delta_t*get_sderivative(F, M, curr + k3, m, I, g)

    curr += (k1 + 2*k2 + 2*k3 + k4)/6

    return curr, t


class EnvPhysicsEngine:
    def __init__(self):
        self.t = 0.0
        self.m = 1.236
        self.I = 0.0133
        self.g = 9.81
        self.curr = np.zeros(6)
        self.delta_t = 0.01
      

    def get_currentState(self):
        return self.curr

    def get_time(self):
        return self.t

    def get_derivative(self, F, M):
        s_dot = get_sderivative(F, M, self.curr, self.m,
                                self.I, self.g)
        return s_dot

    def stepSimulation(self, F, M):
        self.curr, self.t = update(F, M, self.curr, self.t, self.delta_t,
                                   self.m, self.I, self.g)
        pass

    def reset(self, phi, phi_dot):
        self.t = 0.0
        # self.theta_d_prev = phi 
        self.curr[:] = 0.0
        self.curr[2] = phi
        self.curr[5] = phi_dot

    
