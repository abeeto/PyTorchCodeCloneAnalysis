import torch
import numpy as np
from tqdm import tqdm
from network.mlp import MLP
from model.dqn import DQN
from utils.general import get_device


class Tester:
    def __init__(self, env, opts, hyps):
        self.env = env
        self.opts = opts
        self.hyps = hyps
        # extract vars
        self.observation_space = len(self.env.observation_space)
        self.action_space = self.env.action_space.n

        self.device = get_device()

    def setup_model(self):
        policy_net = MLP(self.observation_space, self.action_space)
        self.model = DQN(
            policy_net, device=self.device, hyps=self.hyps, target_net=None
        )
        self.model.exploration_rate = 0.000
        self.model.load(self.opts["model_dir"])

    def setup(self):
        self.setup_model()

    def play(self, initial_state, render=False):
        total_reward = 0
        state = initial_state
        steps = 0
        while True:
            steps += 1
            state = np.reshape(state, [1, self.observation_space])
            if render:
                self.env.render()
            with torch.no_grad():
                action = self.model.get_action(state, training=False)
                state, reward, done, _ = self.env.step(action)
                # print(state, reward, done, action)
                if steps % 1000 == 0:
                    self.env.render = True
                total_reward += reward
                if done:
                    break
        print(total_reward)
        return total_reward

    def run(self):
        totals = []
        num_episodes = 200
        # with tqdm(total=num_episodes) as pbar:
        #     for _ in range(num_episodes):
        #         initial_state = self.env.reset()
        #         ep_reward = self.play(initial_state)
        #         totals.append(ep_reward)
        #         self.env.close()
        #         pbar.update(1)

        # print("Average: {0:.1f}".format(np.mean(totals)))
        # print("Stdev: {0:.1f}".format(np.std(totals)))
        # print("Minumum: {0:.0f}".format(np.min(totals)))
        # print("Maximum: {0:.0f}".format(np.max(totals)))

        # display
        self.play(self.env.reset(), render=True)
