"""
PyTorch MNIST implementation
"""
import time

import torch
from torch.autograd import Variable
from torch.nn import Module, Conv2d, Dropout2d, functional, Linear
from torchvision import datasets, transforms

from weight_gen import get_torch_weights


class Model(Module):
    def __init__(self, weights=None):
        super(Model, self).__init__()
        self.conv1 = Conv2d(1, 10, kernel_size=5)
        self.conv2 = Conv2d(10, 20, kernel_size=5)
        self.drop = Dropout2d()
        self.fc1 = Linear(320, 50)
        self.fc2 = Linear(50, 10)
        self.reset_weights(weights)

    def reset_weights(self, weights):
        if weights is not None:
            self.conv1.weight.data = torch.from_numpy(weights[0])
            self.conv1.bias.data = torch.from_numpy(weights[1])
            self.conv2.weight.data = torch.from_numpy(weights[2])
            self.conv2.bias.data = torch.from_numpy(weights[3])
            self.fc1.weight.data = torch.from_numpy(weights[4])
            self.fc1.bias.data = torch.from_numpy(weights[5])
            self.fc2.weight.data = torch.from_numpy(weights[6])
            self.fc2.bias.data = torch.from_numpy(weights[7])

    def forward(self, x):
        x = functional.relu(functional.max_pool2d(self.conv1(x), 2))
        x = functional.relu(
            functional.max_pool2d(self.drop(self.conv2(x)), 2)
        )
        x = x.view(-1, 320)
        x = functional.relu(self.fc1(x))
        x = functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return functional.log_softmax(x, dim=1)


model = Model(get_torch_weights()).type(torch.FloatTensor).cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(iterations):
    model.train()
    i = 0
    while i < iterations:
        for batch_idx, (data, target) in enumerate(train_loader):
            if i >= iterations:
                break
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Loss at iteration %04d is %.05f' % (i, loss))
            i += 1


kwargs = {'num_workers': 1, 'pin_memory': True, 'batch_size': 64}


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./pt_data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    shuffle=True, drop_last=True, **kwargs)


NUM_ITER = 10000


start = time.time()
train(NUM_ITER)
diff = time.time() - start
print('Time for %d iterations is: %s' % (NUM_ITER, diff))
