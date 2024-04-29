import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable


#######DATEN LADEN
kwargs = {'num_workers': 1, 'pin_memory': True}  # klammer ist leer wenn kein Cuda(grafikkarte) genutz wird
train_data = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('data', train=True, download=True,
                            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                    torchvision.transforms.Normalize((0.1307,),(
                                                                                                        0.3082,))])),
        batch_size=64, shuffle=True, **kwargs)

test_data = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('data', train=False,
                                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                        torchvision.transforms.Normalize((0.1307,),(
                                                                                                        0.3082,))])),
        batch_size=64, shuffle=True, **kwargs)


class MeinNetz(nn.Module):
    def __init__(self):
        super(MeinNetz, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropaut = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv_dropaut(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


model = MeinNetz()
model = model.cuda()

if os.path.isfile('mnistNetz.pt'):
    model = torch.load('mnistNetz.pt')

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)


def train(epoch):
    model.train()  # lernen

    for batch_id, (data, target) in enumerate(train_data):
        data = data.cuda()
        target = target.cuda()
        data = Variable(data)
        target = Variable(target)

        optimizer.zero_grad()
        out = model(data)
        criterion = F.nll_loss
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_id * len(data), len(train_data.dataset), 100. * batch_id / len(train_data), loss.data[0])
        )


def test():
    model.eval()
    loss = 0
    correct = 0
    for data, target in test_data:
        data = Variable(data.cuda(), volatile=True)
        target = Variable(target.cuda())
        out = model(data)
        loss += F.nll_loss(out, target, size_average=False).data[0]
        prediction = out.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()
    loss = loss / len(test_data.dataset)
    print('Durchschnittlicher Loss:', loss)
    print('Erfolge:', 100.*correct/len(test_data.dataset))


if __name__ == '__main__':
    for epoch in range(1, 2):
        train(epoch)
        test()
    torch.save(model, 'mnistNetz.pt')
