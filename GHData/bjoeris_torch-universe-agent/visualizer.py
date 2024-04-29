import gym
import torch
from torch import nn
from torch import multiprocessing
from torch import autograd

class Visualizer:
    def __init__(self,
                 env: gym.Env,
                 model: nn.Module,
                 terminate: multiprocessing.Value=None):
        self.env = env
        self.model = model
        self.terminate = terminate

    def run(self):
        observation = self._observation_tensor(self.env.reset())
        state = self.model.get_initial_state()
        while not self.terminate.value:
            self.env.render()

            log_prob, value, state = self.model(observation, state)
            prob = torch.exp(log_prob)
            action = prob.data.multinomial(1).cpu().numpy()[0]

            observation, _reward, done, _info = self.env.step(action)
            observation = self._observation_tensor(observation)

            if done:
                observation = self._observation_tensor(self.env.reset())
                state = self.model.get_initial_state()
            else:
                state.detach_()

    def _observation_tensor(self, observation):
        observation = torch.FloatTensor(observation)
        if self.model.is_cuda:
            observation = observation.cuda()
        return autograd.Variable(observation)

