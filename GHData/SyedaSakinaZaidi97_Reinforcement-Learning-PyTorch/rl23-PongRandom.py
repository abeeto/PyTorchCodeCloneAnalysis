#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: udemy
"""

#import gym
import torch
import random

import matplotlib.pyplot as plt

from atari_wrappers import make_atari, wrap_deepmind

env_id = "PongNoFrameskip-v4"
env = make_atari(env_id)
env = wrap_deepmind(env)

seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

num_episodes = 5

rewards_total = []

for i_episode in range(num_episodes):
    
    state = env.reset()
    
    score = 0
    #for step in range(100):
    while True:
                
        action = env.action_space.sample()
        
        new_state, reward, done, info = env.step(action)
        
        score += reward
        #print(new_state)
        #print(info)
        
        #env.render()
        
        if done:
            rewards_total.append(score)
            print("Episode finished. Reward: %i" % score )
            break
        

print("Average reward: %.2f" % (sum(rewards_total)/num_episodes))
print("Average reward (last 100 episodes): %.2f" % (sum(rewards_total[-100:])/100))

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.plot(rewards_total, alpha=0.6, color='green')
plt.show()

env.close()
env.env.close()
