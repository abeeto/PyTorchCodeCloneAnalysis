#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

BONUS lesson - solving stochastic FrozenLake with value iteration

@author: udemy
"""


import gym
import torch
import matplotlib.pyplot as plt

# stochastic 
env = gym.make('FrozenLake-v0')

number_of_states = env.observation_space.n
number_of_actions =  env.action_space.n

# V values - size 16 (as number of fields[states] in our lake)
V = torch.zeros([number_of_states])

gamma = 0.9

rewards_total = []
steps_total = []
    
# this is common function use in value_iteration and in build_policy
# it goes through all possible moves from defined state 
# and it returns best possible move (its value and index)
def next_step_evaluation(state,Vvalues):
    Vtemp = torch.zeros(number_of_actions)

    for action_possible in range(number_of_actions):

        for prob, new_state, reward, _ in env.env.P[state][action_possible]:
            Vtemp[action_possible] += (prob * (reward + gamma * Vvalues[new_state]) )
    
    max_value, indice = torch.max(Vtemp,0)
    
    return max_value, indice
    
    
# VALUE ITERATION
# this will build V values table from scratch
# will go through all possible states 
def value_iteration():
    Qvalues = torch.zeros(number_of_states)
    # this is value based on experiments 
    # after that many iterations values don't change significantly any more
    # it can be done in better way - with some kind of evaluation of our values
    # but this is simplified version which works also well in this example
    max_iterations = 1500

    for _ in range(max_iterations):
        # for each step we search for best possible move
        for state in range(number_of_states):
            max_value, _ = next_step_evaluation(state, Qvalues)
            Qvalues[state] = max_value[0]

    return Qvalues

# BUILD POLICY
# NOW having V table - we can use it to build policy 
# policy meaning clear instructions which are best moves from each single state
# So having V values table ready - we can easily understand which move is the best
# in each step
# so we are able to build clear instructions for our agent
# telling him which move he should choose in every state
def build_policy(Vvalues):
    Vpolicy = torch.zeros(number_of_states)

    for state in range(number_of_states):
        _, indice = next_step_evaluation(state, Vvalues)
        Vpolicy[state] = indice[0]
    
    return Vpolicy

# 2 main steps to build policy for our agent
V = value_iteration()
Vpolicy = build_policy(V)


# main loop for our agent
num_episodes = 1000

for i_episode in range(num_episodes):
    state = env.reset()
    
    step = 0 
    while True:
        step += 1
         
        action = Vpolicy[state]

        new_state, reward, done, info = env.step(action)
        
        state = new_state

        if done:
            rewards_total.append(reward)
            steps_total.append(step)
            break

print(V)


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
