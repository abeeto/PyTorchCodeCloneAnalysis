'''
This is a tutorial from Phil Tabor and his YouTube Channel Machine Learning with Phil.  This is the first time I followed
a PyTorch tutorial.  There are some interesting differences between Keras/TensorFlow and PyTorch.  For one, my lack of
familiarity with PyTorch was a bit intimidating but once we got to the building of the neural network, the simplicity
of PyTorch caught my attention immediately.  I am used to working with Keras/TF, which in some cases can be a bit
code heavy.  But with PyTorch, the neural network part was rather readable and easy to put together.  That was the
only specific difference between PyTorch and Keras/TF that I could see.  Everything else that the tutorial covered
is the normal stuff needed to build a DQN agent.

I am looking forward to doing more tutorials with Pytorch.

Phil Tabor:
YouTube: Machine Learning with Phil
Website: neuralnet.ai
Twitter: @MLWithPhil
'''

import gym
import numpy as np
from dueling_dqn_torch import Agent
from utils import plotLearning

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    num_games = 1000
    load_checkpoint = False
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=5e-4, input_dims=[8],
                  n_actions=4, mem_size=1000000, eps_min=0.01, batch_size=64,
                  eps_dec=1e-3, replace=100)

    if load_checkpoint:
        agent.load_models()

    filename = 'lunar_lander_dueling.png'
    scores = []
    eps_history = []
    n_steps = 0

    for i in range(num_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, int(done))
            agent.learn()

            observation = observation_

        scores.append(score)
        avg_score = np.mean(scores[max(0, i-100):(i+1)])  # the previous 100 games to the newest one
        print(f"Your average score is {avg_score} for episode {i}")
        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(num_games)]
    plotLearning(x, scores, eps_history, filename)