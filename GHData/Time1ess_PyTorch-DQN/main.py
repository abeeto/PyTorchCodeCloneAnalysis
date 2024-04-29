import logging
from random import random

import gym
import numpy as np
from tqdm import tqdm

from configs import args
from model import DQNAgent, Statistic
from utils import preprocess, set_global_seed, get_wrapper_by_name
from env import get_env


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=args.log_path,
    filemode='w')
set_global_seed(args.random_seed)

if args.use_wrap_env:
    print('Use wrapped environment.')
    env = get_env(args.env_name, args.random_seed)
else:
    print('Use standard Environment.')
    env = gym.make(args.env_name)
legal_actions = env.action_space
agent = DQNAgent(env, legal_actions.n, args)

observation = env.reset()
total_progress = tqdm(range(1, 1 + args.max_episodes))


def logging_stats(info_dict):
    logging.info(' '.join(
        (key + ':' + val for key, val in sorted(info_dict.items()))))
    total_progress.set_postfix(info_dict)


stat = Statistic(logging_stats, args)
for step in total_progress:
    # Store latest observation
    idx = agent.memory.store_observation(preprocess(observation))

    # Select action with epsilon-greedy policy
    ep = max(0.1, args.epsilon * (1 - step / 1e6))
    if not step > args.learn_start or random() < ep:
        action = env.action_space.sample()
    else:
        history = agent.memory.recent_history()
        action = agent.predict(history)
    observation, reward, done, _ = env.step(action)
    agent.memory.store_effect(idx, action, reward, done)
    if done:
        observation = env.reset()

    result = (0, 0, False)  # Q-value, loss, update
    # Perform experience replay and train the network
    if (step > args.learn_start and step % args.update_freq == 0 and
            len(agent.memory) >= args.batch_size):
        result = agent.q_learn_mini_batch()
        if step % args.target_update_freq == 0:
            agent.update_target_dqn()

    if step % 10000 == 0:
        episode_rewards = get_wrapper_by_name(env,
                                              "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
            print(mean_episode_reward)
    if step % args.save_freq == 0:
        agent.save(step)
    stat.on_step(step, reward, done, *result)
