import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

batch_size = 200
learning_rate = 0.01
epochs = 10

train_db = datasets.MNIST('./data',
                          train=True,
                          download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                          ]))
test_db = datasets.MNIST('./data',
                         train=False,
                         download=False,
                         transform=transforms.Compose([transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_db,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_db,
                                          batch_size=batch_size,
                                          shuffle=True)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),  # inplace 是否将计算得到的值覆盖之前的值
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x


net = MLP()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.01)
criteon = nn.CrossEntropyLoss()


def main():
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 28 * 28)

            logits = net(data)
            loss = criteon(logits, target)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if batch_idx % 100 == 0:
                print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data = data.view(-1, 28 * 28)
            logits = net(data)

            test_loss += criteon(logits, target).item()

            pred = logits.argmax(dim=1)
            correct += pred.eq(target).float().sum().item()

        test_loss /= len(test_loader.dataset)
        print(
            '\ntest set: ave loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    main()