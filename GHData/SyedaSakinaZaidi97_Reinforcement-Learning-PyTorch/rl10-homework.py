#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: udemy
"""


import gym
import time
import torch

import matplotlib.pyplot as plt

env = gym.make('Taxi-v2')

plt.style.use('ggplot')

number_of_states = env.observation_space.n
number_of_actions = env.action_space.n

gamma = 0.95
learning_rate = 0.9
egreedy = 0.1

Q = torch.zeros([number_of_states, number_of_actions])

num_episodes = 1000

steps_total = []
rewards_total = []

for i_episode in range(num_episodes):
    
    state = env.reset()
    
    step = 0
    score = 0
    #for step in range(100):
    while True:
        
        step += 1
        
        #action = env.action_space.sample()
        random_for_egreedy = torch.rand(1)[0]
        
        if random_for_egreedy > egreedy:      
            random_values = Q[state] + torch.rand(1,number_of_actions) / 1000      
            action = torch.max(random_values,1)[1][0]          
        else:
            action = env.action_space.sample()

        
        new_state, reward, done, info = env.step(action)
        
        score += reward

        Q[state, action] = (1 - learning_rate) * Q[state, action] \
            + learning_rate * (reward + gamma * torch.max(Q[new_state]))
        
        state = new_state
        
        #time.sleep(0.4)
        
        #env.render()
        
        #print(new_state)
        #print(info)
        
        if done:
            steps_total.append(step)
            rewards_total.append(score)
            print("Episode finished after %i steps" % step )
            print(score)
            break
        
print(Q)
        
print("Percent of episodes finished successfully: {0}".format(sum(rewards_total)/num_episodes))
print("Percent of episodes finished successfully (last 100 episodes): {0}".format(sum(rewards_total[-100:])/100))

print("Average number of steps: %.2f" % (sum(steps_total)/num_episodes))
print("Average number of steps (last 100 episodes): %.2f" % (sum(steps_total[-100:])/100))

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color='green')
plt.show()

plt.figure(figsize=(12,5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='red')
plt.show()

