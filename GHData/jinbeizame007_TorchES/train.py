import torch
from torch import nn
import numpy as np
import gym
import multiprocessing as mp

from torches.cmaes import CMAES
from models import Policy


def rollout(policy):
    env = gym.make('Pendulum-v0')
    obs = env.reset()
    done = False
    cumulative_reward = 0
    while not done:
        with torch.no_grad():
            action = policy.get_action(obs)
        obs, reward, done, _ = env.step(action*2)
        cumulative_reward += reward
    return cumulative_reward


def fit_func(policies, queue, iterations=2):
    fitness = [-np.average([rollout(policy,) for _ in range(iterations)]) for policy in policies]
    queue.put(fitness)


def run():
    n_process = 16
    popsize = 64
    process_popsize = popsize // n_process

    policy = Policy()
    cmaes = CMAES(policy, popsize=popsize)

    for epoch in range(500):
        policies = cmaes.ask()

        queues = [mp.Queue() for _ in range(n_process)]
        processes = []
        for i in range(n_process):
            processes.append(mp.Process(
                target = fit_func,
                args = (policies[i*process_popsize: (i+1)*process_popsize], queues[i])
            ))
        
        for p in processes:
            p.start()
        
        for p in processes:
            p.join()
        
        fitness = []
        for queue in queues:
            fitness += queue.get()

        cmaes.tell(fitness)
        print('epoch:', epoch, 'mean:', np.average(fitness), 'min:', np.min(fitness))

if __name__ == '__main__':
    run()