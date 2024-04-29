import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


in_dim = 28*28
out_class = 10
batch_size = 64
# 迭代次数 10
epochs_num = 100


class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.logistic = torch.nn.Linear(in_dim, out_class)
    
    def forward(self, x):
        return self.logistic(x)
    
model = LogisticRegression()

# 交叉熵损失
loss_function = torch.nn.CrossEntropyLoss()

# 随机梯度下降，学习率 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 另外一个优化方法 Adam
#optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(data.size(0), -1)
        output = model(data)
        loss = loss_function(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)
            output = model(data)
            test_loss += loss_function(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)



for epoch in range(epochs_num):
    train(model, train_loader, optimizer, epoch)
    test(model, test_loader)

