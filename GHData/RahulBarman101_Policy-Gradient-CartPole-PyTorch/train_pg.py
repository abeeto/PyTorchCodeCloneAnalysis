import numpy as np
import gym
from policy_gradient_pytorch import Agent
import time

agent = Agent([4])
env = gym.make('CartPole-v0')

num_episodes = 1000
j = 0
for i in range(num_episodes):
    state = env.reset()
    score = 0
    done = False
    j += 1
    while not done:
        action = agent.choose_action(state)
        new_state,reward,done,_ = env.step(action)
        agent.store_rewards(reward)
        state = new_state
        score += reward
        if j%100 == 0:
            env.render()
    # time.sleep(2)
    agent.learn()
    print(f'episode done: {i+1}\t score recieved: {score}')
