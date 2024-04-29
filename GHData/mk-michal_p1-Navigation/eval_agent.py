import argparse
import os
import time

import torch
from unityagents import UnityEnvironment

from navigation_agent import Agent


parser = argparse.ArgumentParser()
parser.add_argument('--n-runs', type=int, default=3)
parser.add_argument('--path', default = 'model/weights.pth')
cfg = parser.parse_args()

def run_trained_agent(
    env,
    agent: Agent,
):
    env_info = env.reset(train_mode=False)['BananaBrain']  # reset the environment
    state = env_info.vector_observations[0]
    score = 0
    while True:
        time.sleep(0.05)
        action = agent.act(state)
        # select an action
        env_info = env.step(action)['BananaBrain']  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break


    print("Score: {}".format(score))



if __name__ == '__main__':
    agent = Agent(37, 4, 3)
    agent.qnetwork_local.load_state_dict(torch.load(cfg.path))
    env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64", worker_id=1, seed=1)
    for run in range(cfg.n_runs):
        run_trained_agent(env, agent)
    env.close()