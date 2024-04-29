from random import randint

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable


class ReplayMemory(object):
    def __init__(self, memory_size, history_size, batch_size):
        self.memory_size = memory_size
        self.history_size = history_size
        self.batch_size = batch_size
        self.idx = 0
        self.observation_cnt = 0
        self.observations = np.empty((self.memory_size, 84, 84),
                                     dtype=np.uint8)
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.finals = np.empty(self.memory_size, dtype=np.uint8)

        self.prestates = np.empty([batch_size, history_size, 84, 84],
                                  dtype=np.float32)
        self.poststates = np.empty([batch_size, history_size, 84, 84],
                                   dtype=np.float32)

    def sample(self):
        indices = []
        while len(indices) < self.batch_size:
            while True:
                # Observation index should in range(history_size - 1, cnt - 2)
                # e.g. Minimum: 3 (concatenate [0, 1, 2, 3]
                idx = randint(self.history_size - 1, self.observation_cnt - 2)
                # Cross new and old experience, skip this
                if idx >= self.idx and idx - self.history_size < self.idx:
                    continue
                # Not continuous, skip
                if self.finals[idx - self.history_size + 1:idx + 1].any():
                    continue
                # Already sampled, skip
                if idx in indices:
                    continue
                break
            self.prestates[len(indices)] = self.retrieve(idx)
            self.poststates[len(indices)] = self.retrieve(idx + 1)
            indices.append(idx)
        acts = self.actions[indices]
        rews = self.rewards[indices]
        finals = self.finals[indices]
        return self.prestates, acts, rews, self.poststates, finals

    def recent_history(self):
        assert self.observation_cnt >= self.history_size
        return self.retrieve(self.idx - 1).astype(np.float32)

    def store_observation(self, observation):
        idx = self.idx
        self.observations[idx] = observation
        self.idx = (idx + 1) % self.memory_size
        self.observation_cnt = max(self.observation_cnt, self.idx + 1)
        return idx

    def store_effect(self, idx, action, reward, done):
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.finals[idx] = 1 if done else 0

    def retrieve(self, idx):
        """
        Retrieve experience in the range(idx - history_size + 1, idx + 1)
        """
        if idx >= self.history_size - 1:
            return self.observations[idx - self.history_size + 1:idx + 1]
        else:
            indices = [(idx - i) % self.observation_cnt
                       for i in reversed(range(self.history_size))]
            return self.observations[indices]

    def __len__(self):
        return min(self.observation_cnt, self.memory_size)


class DQN(nn.Module):
    def __init__(self, output_size):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, output_size)

        # initializer
        # init.xavier_uniform(self.conv1.weight)
        # init.constant(self.conv1.bias, 0.1)
        # init.xavier_uniform(self.conv2.weight)
        # init.constant(self.conv2.bias, 0.1)
        # init.xavier_uniform(self.conv3.weight)
        # init.constant(self.conv3.bias, 0.1)
        # init.xavier_uniform(self.fc4.weight)
        # init.constant(self.fc4.bias, 0.1)
        # init.xavier_uniform(self.fc5.weight)
        # init.constant(self.fc5.bias, 0.1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = self.fc5(x)
        return x


class DQNAgent(object):
    def __init__(self, env, output_size, args):
        self.args = args
        self.env = env
        self.memory = ReplayMemory(args.memory_size,
                                   args.history_size,
                                   args.batch_size)
        self.dqn = DQN(output_size)
        self.target_dqn = DQN(output_size)
        self.criterion = nn.SmoothL1Loss()
        if args.use_cuda:
            self.dqn = self.dqn.cuda(args.gpu_idx)
            self.target_dqn = self.target_dqn.cuda(args.gpu_idx)
            self.criterion = self.criterion.cuda(args.gpu_idx)
        self.optimizer = optim.RMSprop(
            self.dqn.parameters(),
            lr=args.lr,
            momentum=args.rmsprop_momentum,
            alpha=args.rmsprop_alpha,
            eps=args.rmsprop_ep)

    def q_learn_mini_batch(self):
        args = self.args
        obs, acts, rews, next_obs, finals = self.memory.sample()

        # To tensors
        observations = torch.from_numpy(obs).div(255.0)
        actions = torch.from_numpy(acts).view(-1, 1).long()
        rewards = torch.from_numpy(rews)
        next_observations = torch.from_numpy(next_obs).div(255.0)
        game_over_mask = torch.from_numpy(finals)
        if args.use_cuda:
            observations = observations.cuda(args.gpu_idx)
            actions = actions.cuda(args.gpu_idx)
            rewards = rewards.cuda(args.gpu_idx)
            next_observations = next_observations.cuda(args.gpu_idx)
            game_over_mask = game_over_mask.cuda(args.gpu_idx)

        # To variables
        observations = Variable(observations)
        actions = Variable(actions)
        rewards = Variable(rewards)
        next_observations = Variable(next_observations, volatile=True)
        game_over_mask = Variable(game_over_mask)

        next_rewards = self.qvalue(next_observations,
                                   use_target=True).max(1)[0]
        next_rewards[game_over_mask] = 0
        target_rewards = (rewards + args.gamma * next_rewards).view(-1, 1)
        target_rewards.volatile = False
        prediction_rewards = self.qvalue(observations).gather(1, actions)
        loss = self.criterion(prediction_rewards, target_rewards)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.dqn.parameters():
            nn.utils.clip_grad_norm(param, 10)
        self.optimizer.step()
        q = prediction_rewards.data.cpu().mean()
        return q, loss.data.cpu()[0], True

    def update_target_dqn(self):
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    def qvalue(self, x, use_target=False):
        return self.target_dqn(x) if use_target else self.dqn(x)

    def predict(self, history):
        args = self.args
        history = torch.from_numpy(history).div_(255.0).view(1, 4, 84, 84)
        if args.use_cuda:
            history = history.cuda(args.gpu_idx)
        var_his = Variable(history, volatile=True)
        action = self.qvalue(var_his).max(1)[1].cpu().data[0]
        return action

    def save(self, step):
        torch.save(self.dqn.state_dict(),
                   'saves/DQN_{}_{}.bin'.format(self.args.env_name, step))


class Statistic(object):
    def __init__(self, output, args):
        self.output = output
        self.args = args
        self.reset()

    def reset(self):
        self.num_game = 0
        self.total_q = 0
        self.total_loss = 0
        self.total_reward = 0
        self.ep_reward = 0
        self.num_updates = 0
        self.ep_rewards = []

    def on_step(self, step, reward, done, q, loss, is_update):
        args = self.args

        if step < args.learn_start:
            return

        self.total_q += q
        self.total_loss += loss
        self.total_reward += reward

        if done:
            self.num_game += 1
            self.ep_rewards.append(self.ep_reward)
            self.ep_reward = 0
        else:
            self.ep_reward += reward

        if is_update:
            self.num_updates += 1

        freq = args.target_update_freq
        if self.num_updates % freq == freq - 1:
            avg_q = self.total_q / self.num_updates
            avg_loss = self.total_loss / self.num_updates
            avg_reward = self.total_reward / self.num_updates
            try:
                max_ep_reward = np.max(self.ep_rewards)
                min_ep_reward = np.min(self.ep_rewards)
                avg_ep_reward = np.mean(self.ep_rewards)
            except Exception as e:
                max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

            self.output({
                'STEP': str(step),
                'AVG_Q': '{:.4f}'.format(avg_q),
                'AVG_L': '{:.4f}'.format(avg_loss),
                'AVG_R': '{:.4f}'.format(avg_reward),
                'EP_MAX_R': '{:.4f}'.format(max_ep_reward),
                'EP_MIN_R': '{:.4f}'.format(min_ep_reward),
                'EP_AVG_R': '{:.4f}'.format(avg_ep_reward),
                'NUM_GAME': '{:.4f}'.format(self.num_game),
            })

            self.reset()
