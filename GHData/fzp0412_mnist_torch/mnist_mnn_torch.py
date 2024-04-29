import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



batch_size = 128
epochs = 10
learning_rate =1e-2
class_num = 10
hide1_num = 512
hide2_num = 256

kwargs = {'num_workers': 1, 'pin_memory': True} if 0 else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self,in_dim,hide1_num,hide2_num,class_num):
        super(Net,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,hide1_num),
                                    nn.BatchNorm1d(hide1_num))
        self.layer2 = nn.Sequential(nn.Linear(hide1_num,hide2_num),
                                    nn.BatchNorm1d(hide2_num))
        self.layer3 = nn.Linear(hide2_num,class_num)

    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return F.log_softmax(x,dim=1)

model = Net(28*28,hide1_num,hide2_num,class_num)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data =data.view(data.shape[0],-1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data =data.view(data.shape[0],-1)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


s_time = int(time.time())
for epoch in range(epochs):
    train(epoch)
    test()
e_time = int(time.time())
print("%02d:%02d:%02d" %((e_time-s_time)/3600,(e_time-s_time)%3600/60,(e_time-s_time)%60))


