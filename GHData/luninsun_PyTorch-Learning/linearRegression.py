import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

batch_size = 200
learning_rate = 0.01
epochs = 10

train_dataset = datasets.MNIST('./data',
                               train=True,
                               download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307, ), (0.3081, ))
                               ]))
test_dataset = datasets.MNIST('./data',
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

w1, b1 = torch.randn(200, 784,
                     requires_grad=True), torch.randn(200, requires_grad=True)
w2, b2 = torch.randn(200, 200,
                     requires_grad=True), torch.randn(200, requires_grad=True)
w3, b3 = torch.randn(10, 200,
                     requires_grad=True), torch.randn(10, requires_grad=True)

torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)


def forward(x):
    x = x @ w1.t() + b1
    x = F.relu(x)
    x = x @ w2.t() + b2
    x = F.relu(x)
    x = x @ w3.t() + b3
    x = F.relu(x)

    return x


optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
criteon = nn.CrossEntropyLoss()


def main():
    for epoch in range(epochs):

        for batch, (data, target) in enumerate(train_loader):

            data = data.view(-1, 28 * 28)

            logits = forward(data)
            loss = criteon(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print('batch: {}, \tloss: {}'.format(batch, loss))

        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data = data.view(-1, 28 * 28)
            logits = forward(data)
            test_loss += criteon(logits, target)

            pred = logits.data.max(1)[1]
            correct += pred.eq(target.data).sum()

        test_loss /= len(test_loader.dataset)
        print('\nepoch: {}, \tavg loss: {}, \tcorrect: {}\n'.format(
            epoch, test_loss, correct / len(test_loader.dataset)))


if __name__ == "__main__":
    main()