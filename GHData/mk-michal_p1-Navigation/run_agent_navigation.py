import datetime
import os

import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from navigation_agent import Agent, AgentHyperparams
from navigation_agent import Agent


class ModelHyperparams:
    EPSILON = 0.8
    N_EPISODES = 2000
    MAX_T = 1000
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 0.995


def dqn(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    timestamp = datetime.datetime.now()
    folder = str(datetime.datetime.now()).replace(' ', '_')

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)['BananaBrain']
        state = env_info.vector_observations[0]

        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)['BananaBrain']
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)),
              end="")
        if i_episode % 100 == 0:
            os.makedirs(os.path.join('models', folder), exist_ok=True)
            torch.save(agent.qnetwork_local.state_dict(), os.path.join('models',folder, f'model_{i_episode}.pth'))
            with open(os.path.join('models', folder, 'results.txt'), 'a+') as f:
                f.write(f'Result for episode {i_episode}: {np.round(np.mean(scores_window), 2)} \n')

            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode - 100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

    return scores


if __name__ == '__main__':

    hpar_model = {a:b for a, b in ModelHyperparams.__dict__.items() if not a.startswith('__')}
    hpar_agent = {a:b for a, b in AgentHyperparams.__dict__.items() if not a.startswith('__')}
    all_params = {'model_hyperparameters': hpar_model, 'agent_hyperparameters': hpar_agent}


    # get the default brain
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64", worker_id=1, seed=1)

    agent = Agent(state_size=37, action_size=4, seed=1)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # plot the scores
    scores = dqn(
        env,
        agent,
        n_episodes=ModelHyperparams.N_EPISODES,
        max_t=ModelHyperparams.MAX_T,
        eps_start=ModelHyperparams.EPS_START,
        eps_end=ModelHyperparams.EPS_END,
        eps_decay=ModelHyperparams.EPS_DECAY
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('results_plot')

