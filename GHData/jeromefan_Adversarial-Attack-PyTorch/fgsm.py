import json
from baseAlgorithm import BaseAlgorithm


class FGSM(BaseAlgorithm):

    def __init__(self, model, loss_fn, std):
        super(FGSM, self).__init__(model, loss_fn, std)
        print('欢迎使用FGSM/FGM算法模块！')
        with open('config.json') as config_file:
            config = json.load(config_file)
        self.normType = config['norm_p']
        self.epsilon = int(config['epsilon']) / 255 / self.std

    def perturb(self, data, target):

        return self.fgsm(data, target, self.epsilon, self.normType)


