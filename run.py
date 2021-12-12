import os
import random
import numpy as np
import time
from env import Drone1DEnv
from pybulletsim import init_simulation, end_simulation

drone, marker = init_simulation(render = True)
env = Drone1DEnv(drone, marker, 0)

weights1 = np.load(os.path.join('model/pi/fc0/kernel:0.npy'))
bias1 = np.load(os.path.join('model/pi/fc0/bias:0.npy'))

weights2 = np.load(os.path.join('model/pi/fc1/kernel:0.npy'))
bias2 = np.load(os.path.join('model/pi/fc1/bias:0.npy'))

weights3 = np.load(os.path.join('model/pi/pi/kernel:0.npy'))
bias3 = np.load(os.path.join('model/pi/pi/bias:0.npy'))


def z_calculation(x, weights, bias):
    x = np.reshape(np.array(x), (len(x),-1))
    z = np.sum(weights*x, axis=0) + bias
    return z


def relu_function(z):
    a = np.maximum(z, 0)
    return a

def predict(x):

    z = z_calculation(x, weights1, bias1)
    a = relu_function(z)
    z = z_calculation(a, weights2, bias2)
    a = relu_function(z)
    z = z_calculation(a, weights3, bias3)
    a = np.tanh(z)

    return a[0]

no = np.pi/2
mo = np.deg2rad(200)

obs = env.reset(obser=[-np.pi/2.1,np.pi/6])
print('initialized with:', [np.degrees(obs[0]*no),np.degrees(obs[1]*mo)])
for i in range(900):
    a = predict(obs)
    obs, reward, done, _ = env.step(a)
    print('state: ', [np.degrees(obs[0]*no), np.degrees(obs[1]*mo)])
    time.sleep(0.01)
    if done:
        print('*******time:', i*0.01)
        break
        



end_simulation()