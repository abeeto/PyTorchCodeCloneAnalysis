from torch.autograd import Variable
from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score

class NnImg2Num(nn.Module):
    def __init__(self):
        super(NnImg2Num, self).__init__()
        # Intialize NeuralNetwork
        in_layer = 28*28
        out_layer = 10
        self.linear1 = nn.Linear(in_layer, in_layer/2)
        self.linear2 = nn.Linear(in_layer/2, out_layer)

    def forward(self, img):
        img = img.float()
        if len(img.data.size()) == 4:
            (_, C, H, W) = img.data.size()
            img = img.view(-1, C*H*W)
        elif len(img.size()) == 2:
            (H, W) = img.data.size()
            img = img.view(1, H*W)
        img = self.linear1(img)
        img = F.sigmoid(img)
        img = self.linear2(img)
        img = F.sigmoid(img)
        return img

    def train(self):
        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.2)

        # Load MNIST
        root = 'torchvision/mnist/'
        download = True
        trans = transforms.Compose([transforms.ToTensor()])
        train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
        batch_size = 256
        train_loader = torch.utils.data.DataLoader(
                         dataset=train_set,
                         batch_size=batch_size,
                         shuffle=True)
        # training
        batch_idx = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            self.optimizer.zero_grad()

            x, target = Variable(x), Variable(NnImg2Num.oneHot(target))
            x_pred = self.forward(x)
            loss = self.loss_function(x_pred, target)
            loss.backward()
            self.optimizer.step()
            if (batch_idx+1)% 100 == 0:
                # print '==>>> batch index: {}, train loss: {:.6f}'.format(batch_idx, loss.data[0])
                print '==>>> batch index: {}'.format(batch_idx+1)
        print '==>>> batch index: {}'.format(batch_idx+1)

    @staticmethod
    def oneHot(target):
        # oneHot encoding
        label = []
        for l in target:
                label.append([1 if i==l else 0 for i in range(10)])
        return torch.FloatTensor(label)
