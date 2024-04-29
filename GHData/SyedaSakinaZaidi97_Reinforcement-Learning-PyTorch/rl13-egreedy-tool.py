#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: udemy
"""

import torch
import matplotlib.pyplot as plt
import math
import gym

env = gym.make('CartPole-v0')
num_episodes = 150

egreedy = 0.9
egreedy_final = 0.02
egreedy_decay = 500

egreedy_prev = egreedy
egreedy_prev_final = egreedy_final
egreedy_prev_decay = 0.999

def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
              math.exp(-1. * steps_done / egreedy_decay )
    return epsilon

egreedy_total = []
egreedy_prev_total = []

steps_total = 0 

for i_episode in range(num_episodes):

    state = env.reset()
    while True:
        
        steps_total += 1

        epsilon = calculate_epsilon(steps_total)
        egreedy_total.append(epsilon)

        action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        
        state = new_state  
        if egreedy_prev > egreedy_prev_final:
            egreedy_prev *= egreedy_prev_decay
            egreedy_prev_total.append(egreedy_prev)
        else:
            egreedy_prev_total.append(egreedy_prev)
            
        if done:
            break
                
plt.figure(figsize=(12,5))
plt.title("Egreedy value")
plt.bar(torch.arange(len(egreedy_total)), egreedy_total, alpha=0.6, color='blue')

plt.figure(figsize=(12,5))
plt.title("Egreedy 2 value")
plt.bar(torch.arange(len(egreedy_prev_total)), egreedy_prev_total, alpha=0.6, color='green')

