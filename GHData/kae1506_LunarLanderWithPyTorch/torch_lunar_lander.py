import torch as T
import torch.optim as O
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import plotLearning
import gym

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_shape, n_actions, fc1_dims, fc2_dims):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = O.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()

        self.device = "cuda: 0" if T.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x

class ReplayBuffer(object):
    def __init__(self, mem_size, input_shape, n_actions):
        self.mem_cntr = 0
        self.mem_size = mem_size
        #input_shape = input_shape[0]
        self.state_memory = np.zeros((mem_size, *input_shape))
        self.new_state_memory = np.zeros((mem_size, *input_shape))
        self.action_memory = np.zeros(mem_size, dtype=np.int64)
        self.terminal_memory = np.zeros(mem_size, dtype=np.bool)
        self.reward_memory = np.zeros(mem_size, dtype=np.int32)

    def remember(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample(self, batch_size):
        #batch = min(self.mem_cntr, self.mem_size)
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class Agent(object):
    def __init__(self, lr, gamma, input_shape, n_actions, mem_size, batch_size,eps, eps_dec, eps_end):
        self.gamma = gamma
        self.eps = eps
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.eps_end = eps_end
        self.batch_size = batch_size

        self.memory = ReplayBuffer(mem_size, input_shape, n_actions)
        self.q_eval = DeepQNetwork(lr, input_shape, n_actions, 512, 1024)

        tns = T.Tensor([1,2,3]).to(self.q_eval.device)
        tns2 = T.Tensor([4,5,6]).to(self.q_eval.device)

        # print(T.argmax(tns), T.max(tns))

    def choose_action(self, state):
        state = T.Tensor(state).to(self.q_eval.device)
        if np.random.random() > self.eps:
            # print("Thinking here")
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()

        else:
            # print("Random here")
            action = np.random.choice(self.action_space)

        # print(action)
        return action

    def learn(self):
        if self.memory.mem_cntr >= self.batch_size:
            self.q_eval.optimizer.zero_grad()

            states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)
            # print(dones.shape)

            states = T.Tensor(states).float().to(self.q_eval.device)
            states_ = T.Tensor(states_).float().to(self.q_eval.device)
            actions = T.Tensor(actions).to(T.int64).to(self.q_eval.device)
            rewards = T.Tensor(rewards).float().to(self.q_eval.device)
            terminal = T.Tensor(dones).to(T.bool).to(self.q_eval.device)

            # print(states.dtype, states_.dtype, rewards.dtype)

            # print(states[terminal])
            batch_indices = np.arange(self.batch_size, dtype=np.int64)
            # print(actions.dtype)
            q_eval = self.q_eval(states)[batch_indices, actions]
            # print(q_eval.shape)
            q_next = self.q_eval(states_)
            q_next = T.max(q_next, dim=1)[0]# .to(self.q_eval.device)
            # print(q_next.shape)
            # print(q_next[terminal])
            q_next[terminal] = 0.0

            # q_target = q_eval.clone().detach().to(self.q_eval.device)

            q_target = rewards + q_next
            loss = self.q_eval.loss(q_target, q_eval)
            loss.backward()
            self.q_eval.optimizer.step()

            self.eps = self.eps-self.eps_dec if self.eps > self.eps_end else self.eps_end

if __name__ == '__main__':
    n_games = 500

    epsilons = []
    scores = []

    env = gym.make('LunarLander-v2')

    agent = Agent(lr=0.00001, gamma=0.995, input_shape=(8,), n_actions=4, mem_size=10_000, batch_size=32, eps=1.0, eps_dec=10e-3, eps_end=0.01)

    for i in range(n_games):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            agent.memory.remember(state, action, reward, state_, done)
            score += reward
            agent.learn()
            state = state_

        scores.append(score)
        epsilons.append(agent.eps)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print("Episode: {}, Score: {}, Avg_Score: {}, Epsilon: {}".format(i, score, avg_score, agent.eps))


    x = [i+1 for i in range(n_games)]
    filename="plt"
    plotLearning(x, scores, epsilons, filename)
