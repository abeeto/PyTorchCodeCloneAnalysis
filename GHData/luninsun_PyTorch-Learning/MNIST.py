import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

batch_size = 200
learning_rate = 0.01
epochs = 10

train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307, ), (0.3081, ))
                               ]))
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307, ), (0.3081, ))
                              ]))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x


net = MLP()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss()


def main():

    for epoch in range(epochs):

        for batch, (data, target) in enumerate(train_loader):
            data = data.view(-1, 28 * 28)

            logits = net(data)
            loss = criteon(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print('batch: {}, \tloss: {}'.format(batch, loss.item()))

        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data = data.view(-1, 28 * 28)

            logits = net(data)

            test_loss += criteon(logits, target).item()

            pred = logits.argmax(dim=1)
            correct += pred.eq(target).float().sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nepoch: {}, \tavg loss: {}, a\tccuracy: {}\n'.format(
            epoch, test_loss, correct / len(test_loader.dataset)))


if __name__ == "__main__":
    main()