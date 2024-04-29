import torch
import json
from baseAlgorithm import BaseAlgorithm


class PGD(BaseAlgorithm):

    def __init__(self, model, loss_fn, std):
        super(PGD, self).__init__(model, loss_fn, std)
        print('欢迎使用PGD/BIM/I-FGSM算法模块！')
        with open('config.json') as config_file:
            config = json.load(config_file)
        epsilon = int(config['epsilon'])
        self.epsilon = epsilon / 255 / self.std
        self.num_iter = min(epsilon + 4, int(1.25 * epsilon))
        self.alpha = self.epsilon / self.num_iter

    def perturb(self, data, target):
        x_adv = data.detach().clone()
        for i in range(self.num_iter):
            x_adv = self.fgsm(x_adv, target, self.alpha, 'inf')
            x_adv = torch.clamp(data + torch.clamp(x_adv - data, -self.epsilon, self.epsilon), 0, 1)
        return x_adv
