import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


import numpy as np

from layer.dense import Dense


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.d1 = Dense(784, 1024)
        self.d2 = Dense(1024, 1024)
        self.d3 = Dense(1024, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = self.d3(x)
        #return F.log_softmax(x)
        return x

    def summary(self):
        n_params = 0
        for param in self.parameters():
            print(type(param.data), param.size())
            n_params+=param.numel()

        print('total parameters : ', n_params)


def train_example():
    batch_size = 100
    n_epoch = 10
    lr = 0.0001
    GPU_ID = 1


    model = Net()

    print(' --- model summary ---')
    model.summary()

    # for name, param in model.state_dict().items():
    #     print(name, param)



    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train = True, download= True,
                       transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size = batch_size, shuffle = True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=False, **kwargs
    )


    print(' --- data summary ---')
    print('n_train : ', len(train_loader.dataset))
    print('n_test : ', len(test_loader.dataset))
    print('len_loader : ',len(train_loader))
    print(' ---------------------')

    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum= 0.9)


    for e in range(n_epoch):
        model.train()
        train_loss = 0
        train_acc = 0
        t_begin = time.time()
        for batch_idx, (data,target) in enumerate(train_loader):
            # print(type(data))
            # print(type(target))
            # print(data.size())
            # print(data.view(-1, data.size()[1] * data.size()[2] * data.size()[3]).size())
            # print(target.size())

            data, target = Variable(data).cuda(), Variable(target).cuda()
            optimizer.zero_grad()
            output = model(data)
            batch_loss = F.cross_entropy(output, target)
            batch_loss.backward()
            optimizer.step()
            train_loss +=batch_loss.data[0]
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        t_end = time.time()
        print('Epoch : ',e+1)
        print('time : ', t_end - t_begin)
        print('train loss : ', batch_size * train_loss / len(train_loader.dataset), ' | acc : ', train_acc / len(train_loader.dataset))

        model.eval()
        test_loss = 0
        test_acc = 0

        for data, target in test_loader:
            data, target = Variable(data).cuda(), Variable(target).cuda()
            output = model(data)
            test_loss+=F.cross_entropy(output, target).data[0]
            pred = output.data.max(1, keepdim = True)[1]
            test_acc += pred.eq(target.data.view_as(pred)).cpu().sum()

        print('test loss : ', batch_size * test_loss / len(test_loader.dataset), ' | acc : ', test_acc / len(test_loader.dataset))

    torch.save(model.state_dict(), 'models/sample/mnist.pt')



    return 0


def load_and_test():
    batch_size = 100
    model = Net()

    #before load
    print(' --- model summary ---')
    model.summary()

    model.load_state_dict(torch.load('models/sample/mnist.pt'))

    kwargs = {'num_workers': 1, 'pin_memory': True}
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/mnist', train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs
    )

    print('data load done')
    model = model.cuda()
    model.eval()
    test_loss = 0
    test_acc = 0

    print('begin test')
    for data, target in test_loader:
        data, target = Variable(data).cuda(), Variable(target).cuda()
        output = model(data)
        test_loss+=F.cross_entropy(output, target).data[0]
        pred = output.data.max(1, keepdim = True)[1]
        test_acc += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('test loss : ', batch_size * test_loss / len(test_loader.dataset), ' | acc : ', test_acc / len(test_loader.dataset))



if __name__ == '__main__':
    train_example()
    print('------------training done--------------')
    load_and_test()
