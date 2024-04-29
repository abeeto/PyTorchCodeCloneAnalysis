import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

batch_size = 64

train_dataset = datasets.MNIST(root = './data/', train = True, transform = transforms.ToTensor(), download = False)

test_dataset = datasets.MNIST(root = './data/', train = False, transform = transforms.ToTensor())

train_loader = DataLoader(datasets.MNIST('./data', train = True, download = False, transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size = batch_size, shuffle = True)

test_loader = DataLoader(datasets.MNIST('./data', train = False, download = False, transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size = batch_size, shuffle = True)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input size = 1 * 32 * 32
        self.Conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = 5)  # 10 * 28 * 28
        self.Conv2 = torch.nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = 5)  # 20 * 24 * 24
        self.Pooling = torch.nn.MaxPool2d(kernel_size = 2)  # 20 * 12 * 12
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        x_size = x.size(0)
        out1 = F.relu(self.Pooling(self.Conv1(x)))
        out2 = F.relu(self.Pooling(self.Conv2(out1)))
        out3 = out2.view(x_size, -1)
        y_pred = self.fc(out3)
        return F.log_softmax(y_pred)


model = CNN()
print(model.Conv1, '\n', model.Conv2, '\n', model.Pooling, '\n', model.fc)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)


def train(epoch):
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_index % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_index * len(data),
                                                                           len(train_loader.dataset),
                                                                           100 * batch_index / len(train_loader),
                                                                           loss.data[0]))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = Variable(data, volatile = True)
        target = Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average = False).data[0]
        pred = output.data.max(1, keepdim = True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                 len(test_loader.dataset),
                                                                                 100 * correct / len(
                                                                                     test_loader.dataset)))


if __name__ == '__main__':
    for epoch in range(1, 10):
        train(epoch)
        test()
