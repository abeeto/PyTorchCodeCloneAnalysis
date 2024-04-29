# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:26:26 2020

@author: coldhenry
"""
import gym  # open ai gym
import time
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('FrozenLake-v0')

def epsilon_greedy(eps, state, Q_func):
    
    if np.random.uniform(0, 1) < eps:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q_func[state])
        
    return action

def softmax_policy(state, Q_func):
    
    val = np.zeros([1,4], dtype=float)
    tau = 1
    for action in range(4):
        val[0][action] = np.exp(Q_func[state, action]/tau)
    
    prob = val[0] / np.sum(val[0])
    
    action = np.random.choice([0,1,2,3], p = prob)
    
    return action


def TestPolicy(Q_func):
    
    count = 0
    trials = 500
     
    for t in range(trials):
        observation = env.reset() 
        #env.render()
        done = False
        while not done:
            action = np.argmax(Q_func[observation])
            observation, reward, done, info = env.step(action)
        if observation == 15:
            count += 1
        
    print("sucess rate", count/trials)
    return count/trials
    

def Q_learning(alpha, gamma, policy='epsilon'):
    
    Q_func = np.zeros((16,4))
    
    trials = 5000
    rate_record = []
    for ep in tqdm(range(trials)):
        env.reset()
        state = 0
        eps = (1 - ep/5000)
        done = False
        while not done:
            #env.render()
            if policy == 'epsilon':
                action = epsilon_greedy(eps, state, Q_func)
            elif policy == 'fixed-epsilon':
                eps = 0.6
                action = epsilon_greedy(eps, state, Q_func)
            elif policy == 'softmax':
                action = softmax_policy(state, Q_func)
            elif policy == 'greedy':
                action = np.argmax(Q_func[state])
                
            next_state, reward, done, info = env.step(action)
            
            new_val = alpha * (reward + gamma * np.max(Q_func[next_state]) - Q_func[state, action])
            Q_func[state, action] += new_val
            
            state = next_state
        
        if ep % 100 == 0:
            rate_record.append(TestPolicy(Q_func))

    return Q_func, rate_record

def plotting(plotSet, fixed):
    
    trials = 5000
    t = np.arange(0, trials, 100)
    plt.figure(figsize=(9, 9))
    
    plt.plot(t, plotSet)
    #plt.plot(t, plotSet[0], t, plotSet[1], t, plotSet[2], '-r')
    #plt.plot(t, plotSet[0], '-r', t, plotSet[1], t, plotSet[2], t, plotSet[3])
    
    #plt.legend(["g = 0.9","g = 0.95","g = 0.99"], fontsize=15)
    #plt.legend(["a = 0.05","a = 0.1","a = 0.25","a = 0.5"], fontsize=15)
    
    plt.xlabel('Episodes', fontsize=15)
    plt.ylabel('Sucess Rate', fontsize=15)
    
    plt.title("Q Learning: softmax {}".format(fixed), fontsize = 20)
    
    plt.savefig("softmax-{}-correct.png".format(fixed))
    plt.show()


if __name__ == '__main__':
    
    alphaSet = [0.05, 0.1, 0.25, 0.5]
    gamma = 0.99
    
    plotSet = []
    for alpha in alphaSet:
        Q_func, rate_record = Q_learning(alpha, gamma)
        plotSet.append(rate_record)
    plotting(plotSet, gamma)

    alpha = 0.05
    gammaSet = [0.9, 0.95, 0.99]
    plotSet = []
    for gamma in gammaSet:
        Q_func, rate_record = Q_learning(alpha, gamma)
        plotSet.append(rate_record)
    plotting(plotSet, alpha)
    
    gamma = 0.99
    alpha = 0.1
    
    Q_func, rate_record = Q_learning(alpha, gamma, 'softmax')
    plotting(rate_record, 0.6)
    
    