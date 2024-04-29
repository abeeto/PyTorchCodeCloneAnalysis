import numpy as np
import random
import math
import matplotlib.pyplot as plt
import argparse
import os
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.autograd as autograd

from replay_buffer import ReplayBuffer

from wrapper.wrapper_mario import wrap_mario
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

parser = argparse.ArgumentParser(description='DQN Mario')
parser.add_argument('--save', dest='save', action='store', default="basic", help='Select save file path')
parser.add_argument('--load', dest='load', action='store', default=None, help='If Load model')
parser.add_argument('--istest', dest='istest', action='store_true', help='If test model')
args = parser.parse_args()

env = gym_super_mario_bros.make('SuperMarioBros-v2')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
env = wrap_mario(env)

observation_size = env.observation_space.shape
action_size = env.action_space.n
print(observation_size)
print(action_size)


USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
device = torch.device("cuda" if USE_CUDA else "cpu")

IS_RENDER = False
loadfile = args.load
TRAIN_START = 500
TARGET_UPDATE = 200
SAVE_PATH = "./save/" + args.save + "/"
FIG_PATH = "./result/" + args.save + "/"


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(action_size)
        return action


def plot(frame_idx, rewards, losses, figpath):
    fig = plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('step %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    fig.savefig(figpath, dpi=fig.dpi)


def compute_loss(model, target, optimizer, replay_buffer, batch_size, gamma):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = model(state)
    next_q_values = target(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    # MSE loss
    # loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
    # Huber loss
    loss = F.smooth_l1_loss(q_value, Variable(expected_q_value.data))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def main():
    print("torch version", torch.__version__)
    agent = DQN(observation_size, action_size).to(device)
    target_net = DQN(observation_size, action_size).to(device)

    optimizer = optim.RMSprop(agent.parameters())
    replay_buffer = ReplayBuffer(10000)

    losses = []
    rewards = []
    episode = 1
    episode_reward = 0
    s = 0
    state = env.reset()
    epsilon_load = 0

    # logger
    wandb.init()
    wandb.config.update(args)
    wandb.watch(agent, log='parameters')

    # Load
    if loadfile is not None:
        loadpath = SAVE_PATH + "mario_" + loadfile + ".pth"
        checkpoint = torch.load(loadpath)

        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        losses = checkpoint['losses']
        rewards = checkpoint['rewards']
        s = checkpoint['step']
        epsilon_load = checkpoint['epsilon']
        episode = checkpoint['episode']
        episode_reward = checkpoint['episode_reward']
        state = checkpoint['state']

    target_net.load_state_dict(agent.state_dict())
    target_net.eval()

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000

    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
        -1. * frame_idx / epsilon_decay) + epsilon_load

    for step in range(s, 100000):
        epsilon = epsilon_by_frame(step)

        action = agent.act(state, epsilon)
        next_state, reward, done, info = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        episode_reward += reward
        state = next_state

        if done:
            print("Episode {}   step {}".format(episode, step))
            state = env.reset()
            rewards.append(episode_reward)
            episode_reward = 0
            episode += 1

        if IS_RENDER:
            env.render()

        if len(replay_buffer) > TRAIN_START:
            if step % TARGET_UPDATE == 0:
                target_net.load_state_dict(agent.state_dict())
            loss = compute_loss(agent, target_net, optimizer, replay_buffer, 32, 0.99)
            losses.append(loss)

            # save model
            save_path = SAVE_PATH + "mario_" + args.save + ".pth"
            checkpoint = {
                            'step': step,
                            'epsilon': epsilon,
                            'episode': episode,
                            'episode_reward': episode_reward,
                            'model_state_dict': agent.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'losses': losses,
                            'rewards': rewards,
                            'state': state,
                         }
            try:
                torch.save(checkpoint, save_path)
            except:
                os.makedirs(SAVE_PATH)
                torch.save(checkpoint, save_path)

            if step % 1000 == 0:
                step_save_path = SAVE_PATH + "mario_" + args.save + "_" + str(step) + ".pth"
                torch.save(checkpoint, step_save_path)

        if step % 5000 == 0:
            try:
                plot(step, rewards, losses, FIG_PATH+"result.png")
            except:
                os.makedirs(FIG_PATH)
                plot(step, rewards, losses, FIG_PATH + "result.png")
            # wandb.log({'step': step, 'losses': losses, 'rewards': rewards})

    env.close()


def test():
    agent = DQN(observation_size, action_size).to(device)
    loadpath = SAVE_PATH + "mario_" + loadfile + ".pth"
    checkpoint = torch.load(loadpath)
    agent.load_state_dict(checkpoint['model_state_dict'])

    state = env.reset()
    episode = 1
    rewards = 0

    for step in range(10000):
        action = agent.act(state, 0)
        next_state, reward, done, info = env.step(action)
        env.render()
        rewards += reward

        if done:
            print("Episode {}   rewards {}".format(episode, rewards))
            env.reset()
            episode += 1
            rewards = 0


if __name__ == "__main__":
    if args.istest:
        test()
    else:
        main()
