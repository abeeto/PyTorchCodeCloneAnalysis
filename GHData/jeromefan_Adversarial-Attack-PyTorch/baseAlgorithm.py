import torch


class BaseAlgorithm:
    def __init__(self, model, loss_fn, std):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        std = std.to(device)
        self.model = model
        self.loss_fn = loss_fn
        self.std = std

    def fgsm(self, data, target, epsilon, normType):

        x_adv = data.detach().clone()
        x_adv.requires_grad = True
        grad = torch.autograd.grad(outputs=self.loss_fn(self.model(x_adv), target), inputs=x_adv)[0]
        if normType == 'inf':
            grad = grad.sign()
        elif normType == '2':
            for i in range(grad.shape[0]):
                norm_2 = torch.norm(grad[i], p=2)
                grad[i] = grad[i] / norm_2
        else:
            raise Exception("请检查您在config.json中输入的范数值，FGSM为inf，FGM为2！")

        x_adv = x_adv + epsilon * grad
        return x_adv
