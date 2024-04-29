# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:55:50 2020

@author: coldhenry
"""
import gym  # open ai gym
import math
import time
import numpy as np
import random
import pickle
from matplotlib import pyplot as plt

distance = [-1, +4, +1, -4]

env = gym.make('FrozenLake-v0')

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def TestPolicy(policy):
    
    count = 0
    trials = 100
     
    for t in range(trials):
        observation = env.reset() 
        #env.render()
        done = False
        while not done:
            action = policy[observation]
            observation, reward, done, info = env.step(action)
        if observation == 15:
            count += 1
        
    print("sucess rate", count/trials)
    return count/trials
    
def LearnModel():
    
    ## sample for 10M times
    record = np.zeros([16,4,16])
    trials = 100000
    t_start = time.time()
    for i_episode in range(trials):
        observation = env.reset() 
        for t in range(50):
            #env.render()
            action = random.randint(0,3)
            prev_obs = observation
            observation, reward, done, info = env.step(action)
            record[prev_obs][action][observation] += 1
            if done:
                #print("Episode finished after {} timesteps".format(t+1))
                break
    print("elapsed time", time.time()-t_start)
    
    ## calculate transition probabilities
    transition_prob = {}
    for state in range(16):
        for action in range(4):
            count = np.zeros([1,16])
            for i in range(4):
                next_state = min(15, max(0, state + distance[i]))
                count[0][next_state] += record[state][action][next_state]
            if np.sum(count):
                count = count / np.sum(count)
            transition_prob[state, action] = count.tolist()
    
    ## define reward function 
    reward_func = {}
    for state in range(16):
        for action in range(4):
            count = np.zeros([1,16])
            for i in range(4):
                next_state = min(15, max(0, state + distance[i]))
                if next_state > state:
                    count[0][next_state] = 10
                elif next_state is state:
                    count[0][next_state] = 0
                else:
                    count[0][next_state] = -10
            reward_func[state, action] = count.tolist()
    
    save_obj(transition_prob, 'trans_prob')        
    save_obj(reward_func, 'reward_func')
    
    return transition_prob, reward_func
 
    
def updateValue(state, policy, gamma, valFunc, reward_func, transition_prob):
    
        value = 0
        neighbor = env.env.P
        for i in range(len(neighbor[state][policy[state]])):
            #print([state, policy[state], i])
            prob = neighbor[state][int(policy[state])][i][0]
            next_state = neighbor[state][policy[state]][i][1]
            reward = neighbor[state][policy[state]][i][2]
            
            
            #next_state = min(15, max(0, state + distance[i]))
            #prob = np.array(env.env.P[state][policy[state]][state][0])
            #prob = transition_prob[(state, policy[state])][0][next_state]
            #reward = reward_func[(state, policy[state])][0][next_state]
            
            value += prob * (reward + gamma * valFunc[next_state])
            #print([prob, reward, next_state, valFunc[next_state], value])
            
        return value
    
    
def PolicyEval(value, gamma, policy, reward_func, transition_prob):

    print("===Policy Evaluation====")
    
    delta = 1
    count = 0
    while delta >= 0.0001:
        delta = 0
        count += 1
        for state in range(16):
            v = value[state]
            value[state] = updateValue(state, policy, gamma, value, reward_func, transition_prob)
            delta = max(delta, abs(v - value[state]))
            
        if count % 10 == 0:
            #([v, value[state]])
            print(delta)
    
    return value, policy 

def findAction(state, gamma, value_func, reward_func, transition_prob):
    
    data = np.zeros([1,4])
    neighbor = env.env.P
    for action in range(4):
        value = 0
        for i in range(len(neighbor[state][action])):    
            prob = neighbor[state][action][i][0]
            next_state = neighbor[state][action][i][1]
            reward = neighbor[state][action][i][2]
            
            #prob = transition_prob[(state, action)][0][next_state]
            #reward = reward_func[(state, action)][0][next_state]
            value += prob * (reward + gamma * value_func[next_state])

        data[0][action] = value
    
    return max(data[0]), np.argmax(data)

def PolicyIter(reward_func, transition_prob):
    
    gamma = 0.99 # discounted factor
    
    # value array initialization
    value_func = np.zeros(16)
    
    # policy initialization
    policy = np.zeros(16)
    
    rate_record = []
    
    finished = True
    for _ in range(50):
        
        value_func, policy = PolicyEval(value_func, gamma, policy, reward_func, transition_prob)
    
        print("===Policy Improvement====")
        for state in range(16):
            old_action = policy[state]
            _, policy[state] = findAction(state, gamma, value_func, reward_func, transition_prob)
            if old_action != policy[state]:
                finished = False
                
        rate_record.append(TestPolicy(policy.astype('int')))
        
        if finished is True:
            break
            
    return value_func, policy, rate_record

def ValueIter(reward_func, transition_prob):
    
    gamma = 0.99 # discounted factor
    
    # value array initialization
    value_func = np.zeros(16)
    
    # policy initialization
    policy = np.zeros(16)
    
    rate_record = []
    
    delta = 1
    count = 0
    while delta > 0.0001:
        delta = 0
        count += 1
        for state in range(16):
            v = value_func[state]
            
            max_val, _ = findAction(state, gamma, value_func, reward_func, transition_prob)
            value_func[state] = max_val
            
            delta = max(delta, abs(v - value_func[state]))
            
        if count % 10 == 0:
            #([v, value[state]])
            print(delta)
            
        for state in range(16):        
            _, policy[state] = findAction(state, gamma, value_func, reward_func, transition_prob)
            
        rate_record.append(TestPolicy(policy.astype('int')))
            
        if count == 50:
            break
            
    return value_func, policy, rate_record

def plotting(plotSet):
    
    t = np.arange(0, 50, 1)
    plt.figure(figsize=(9, 9))
    plt.plot(t, plotSet)
    #plt.legend(["g = 0.9","g = 0.95","g = 0.99"], fontsize=15)
    plt.xlabel('Episodes', fontsize=15)
    plt.ylabel('Sucess Rate', fontsize=15)
    plt.title("Value Iteration", fontsize = 20)
    
    plt.savefig("VI.png")
    plt.show()
    

#%%
if __name__ == '__main__':

    # Question 1-3 
    policy = np.empty([16], dtype=int)
    for i in range(16):
        policy[i] = (i+1) % 4
    
    TestPolicy(policy)
    
    #transition_prob, reward_func = LearnModel()          
    transition_prob = load_obj('trans_prob')
    reward_func = load_obj('reward_func')
    
    ## Policy iteration
    # t_start = time.time()
    # value_func, policy, rate_record = PolicyIter(reward_func, transition_prob)
    
    ## Value iteration
    t_start = time.time()
    value_func, policy, rate_record = ValueIter(reward_func, transition_prob)
    
    plotting(rate_record)
    
    print("elapsed time", time.time()-t_start)
    
    env.close()