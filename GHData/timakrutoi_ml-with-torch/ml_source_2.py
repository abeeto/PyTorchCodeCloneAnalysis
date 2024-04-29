import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from torch.autograd import Variable
from torchvision.datasets import MNIST


class net(nn.Module):

    def __init__(self, num_fe_layers=2, num_class=2):
        super(net, self).__init__()

        # 1. Dummy
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.lc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64)
        x = self.lc(x)

        return x


if __name__ == "__main__":

    lr = 0.01  # learning rate
    momentum = 0.9
    batch_size = 100

    model = net()
    criterion = nn.CrossEntropyLoss()  # loss
    optimizer = optim.SGD(net.parameters(model), lr, momentum)  # optimizer
    # print(model)

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    train_set = MNIST(root='./data', train=True, download=True, transform=trans)
    test_set = MNIST(root='./data', train=False, download=True, transform=trans)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False)

    print('==>>> total training batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))

    for epoch in range(1):
        for batch_idx, (x, target) in enumerate(train_loader):  # reading train data
            optimizer.zero_grad()
            x, target = Variable(x), Variable(target)
            # one hot encode
            y = model(x)
            # print(y.data.size())
            loss = criterion(y, target)  # loss
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('==>>> epoch: {}, index: {}'.format(epoch, batch_idx))

        # ==================================================================
        # Testing
        total_cnt = 0
        correct_cnt = 0
        for batch_idx, (x, target) in enumerate(test_loader):  # reading test data
            y = model(x)
            predict = torch.max(y.data, 1)
            loss = criterion(y, target)

            # what
            _, pred_label = torch.max(y.data, 1)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()

            if batch_idx % 100 == 0:
                print('==>>> epoch: {}, index: {}, acc: {:.3f}, correct: {}, total: {}'.format(
                    epoch, batch_idx, correct_cnt * 0.1 / total_cnt, correct_cnt, total_cnt))

    print("Done!")
