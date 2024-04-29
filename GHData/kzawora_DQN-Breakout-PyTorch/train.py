from typing import Any
from random import sample, random
import gym
from dataclasses import dataclass
from environment import BreakoutFrameStackingEnv
from models import DQN
import torch
from tqdm import tqdm
import wandb
from time import time
import numpy as np
import torch.nn.functional as F


@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer_size = buffer_size
        self.buffer = [None] * buffer_size
        self.idx = 0

    def insert(self, sars):
        self.buffer[self.idx % self.buffer_size] = sars
        self.idx += 1

    def sample(self, num_samples):
        assert num_samples < min(self.idx, self.buffer_size)
        if self.idx < self.buffer_size:
            return sample(self.buffer[:self.idx], num_samples)
        return sample(self.buffer, num_samples)


def update_target_model(model, target):
    target.load_state_dict(model.state_dict())


def train_step(model, state_transitions, target, num_actions, device, gamma=0.99):
    curr_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions])).to(device)
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions])).to(device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).to(device)
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions])).to(device)
    actions = [s.action for s in state_transitions]
    with torch.no_grad():
        qvals_next = target(next_states).max(-1)[0]

    model.opt.zero_grad()
    qvals = model(curr_states)
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)

    loss_fn = torch.nn.SmoothL1Loss()
    loss = loss_fn(torch.sum(qvals * one_hot_actions, -1), rewards.squeeze() + mask[:, 0] * qvals_next * 0.99)
    loss.backward()
    model.opt.step()
    return loss


def main(test=False, checkpoint=None, device='cuda'):
    if not test:
        wandb.init(project='dqn-breakout', name='test3')
    memory_size = 100000
    min_rb_size = 20000
    sample_size = 100
    lr = 0.0001

    eps_min = 0.05
    eps_decay = 0.99995
    env_steps_before_train = 10
    tgt_model_update = 5000

    env = gym.make('Breakout-v0')
    env = FrameStackingAndResizingEnv(env, 84, 84, 4)
    last_observation = env.reset()

    model = DQN(env.observation_space.shape, env.action_space.n, lr=lr).to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    target = DQN(env.observation_space.shape, env.action_space.n).to(device)
    update_target_model(model, target)
    replay = ReplayBuffer(memory_size)
    steps_since_train = 0
    epochs_since_tgt = 0
    step_num = -1 * min_rb_size
    episode_rewards = []
    rolling_reward = 0

    tq = tqdm()
    while True:
        if test:
            env.render()
            time.sleep(0.05)
        tq.update(1)
        eps = max(eps_min, eps_decay ** (step_num))
        if test:
            eps = 0

        if random() < eps:
            action = env.action_space.sample()
        else:
            x = torch.Tensor(last_observation).unsqueeze(0).to(device)
            action = model(x).max(-1)[-1].item()

        observation, reward, done, info = env.step(action)
        rolling_reward += reward
        reward = reward * 0.1
        replay.insert(Sarsd(last_observation, action, reward, observation, done))
        last_observation = observation
        if done:
            episode_rewards.append(rolling_reward)
            if test:
                print(rolling_reward)
            rolling_reward = 0
            observation = env.reset()
        steps_since_train += 1
        step_num += 1
        if (not test) and (replay.idx > min_rb_size) and (steps_since_train > env_steps_before_train):
            loss = train_step(model, replay.sample(sample_size), target, env.action_space.n, device)
            wandb.log(
                {
                    "loss": loss.detach().cpu().item(),
                    "eps": eps,
                    "avg_reward": np.mean(episode_rewards)
                }
            )
            episode_rewards = []
            epochs_since_tgt += 1
            if epochs_since_tgt > tgt_model_update:
                print('updating target model')
                update_target_model(model, target)
                epochs_since_tgt = 0
                torch.save(target.state_dict(), f'target.model')
            steps_since_train = 0
    env.close()


if __name__ == '__main__':
    main()
