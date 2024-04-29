import torch
import json
from baseAlgorithm import BaseAlgorithm


class MIFGSM(BaseAlgorithm):

    def __init__(self, model, loss_fn, std):
        super(MIFGSM, self).__init__(model, loss_fn, std)
        with open('config.json') as config_file:
            config = json.load(config_file)
        self.normType = config['norm_p']
        epsilon = int(config['epsilon'])
        self.epsilon = epsilon / 255 / self.std
        self.num_iter = min(epsilon + 4, int(1.25 * epsilon))
        self.alpha = self.epsilon / self.num_iter
        self.g = 0
        self.miu = float(config['miu'])

    def perturb(self, data, target):
        x_adv = data.detach().clone()
        for i in range(self.num_iter):
            x_adv = self.fgsm(x_adv, target, self.alpha, self.normType)
            x_adv = torch.min(torch.max(x_adv, data - self.epsilon), data + self.epsilon)
        return x_adv

    def fgsm(self, data, target, epsilon, normType):

        x_adv = data.detach().clone()
        x_adv.requires_grad = True
        grad = torch.autograd.grad(outputs=self.loss_fn(self.model(x_adv), target), inputs=x_adv)[0]
        for i in range(grad.shape[0]):
            norm_1 = torch.norm(grad[i], p=1)
            grad[i] = grad[i] / norm_1
        self.g = self.miu * self.g + grad

        gd = self.g.clone()
        if normType == 'inf':
            gd = gd.sign()
        elif normType == '2':
            for i in range(gd.shape[0]):
                norm_2 = torch.norm(gd, p=2)
                gd = gd / norm_2
        else:
            raise Exception("请检查您在config.json中输入的范数值，MI-FGSM仅支持 inf或2！")

        x_adv = x_adv + epsilon * gd
        return x_adv
