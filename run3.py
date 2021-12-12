import os
from pickle import FALSE
import numpy as np
import random
import time
from env_pybullet import Drone1DEnv
from pybulletsim import init_simulation, end_simulation

drone, marker = init_simulation(render = True)
env = Drone1DEnv(drone, marker, np.deg2rad(30.0))

weights1 = np.load(os.path.join('pi/fc0/kernel:0.npy'))
bias1 = np.load(os.path.join('pi/fc0/bias:0.npy'))

weights2 = np.load(os.path.join('pi/fc1/kernel:0.npy'))
bias2 = np.load(os.path.join('pi/fc1/bias:0.npy'))

weights3 = np.load(os.path.join('pi/pi/kernel:0.npy'))
bias3 = np.load(os.path.join('pi/pi/bias:0.npy'))


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



obs = env.reset(obser=[np.deg2rad(0),np.deg2rad(0)])
print('initialized with:', np.degrees([obs[0]/1.5, obs[1]/(8/0.523)]))
for i in range(1000):
    a = predict(obs)
    obs, done = env.step(a)
    print('state: ', np.degrees(obs[1]/16))
    time.sleep(0.01)
    if done:
        print('*******time:', i*0.01)
        break
        



end_simulation()