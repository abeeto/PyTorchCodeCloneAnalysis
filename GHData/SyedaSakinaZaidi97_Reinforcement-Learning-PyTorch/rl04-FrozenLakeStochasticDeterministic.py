#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: udemy
"""


import gym
import time

import matplotlib.pyplot as plt

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
)

env = gym.make('FrozenLakeNotSlippery-v0')

num_episodes = 1000

steps_total = []

for i_episode in range(num_episodes):
    
    state = env.reset()
    
    step = 0
    #for step in range(100):
    while True:
        
        step += 1
        
        action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)
        

        
        time.sleep(0.4)
        
        env.render()
        
        print(new_state)
        print(info)
        
        if done:
            steps_total.append(step)
            print("Episode finished after %i steps" % step )
            break
        
        
print("Average number of steps: %.2f" % (sum(steps_total)/num_episodes))
plt.plot(steps_total)
plt.show()