import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional
from torch import optim
import torchvision
from torch.utils import data


def plot_curve(data_set):
    fig = plt.figure()
    plt.plot(range(len(data_set)), data_set, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(img[i][0] * 0.3081 + 0.1307, cmap='gray', interpolation='none')
        plt.title("{}: {}".format(name, label[i].item))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def ont_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out


# step1: load dataset
train_loader = data.DataLoader(
    torchvision.datasets.MNIST(
        'minst_data',
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]
        )
    ),
    batch_size=512,
    shuffle=True
)

test_loader = data.DataLoader(
    torchvision.datasets.MNIST(
        'minst_data',
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]
        )
    ),
    batch_size=512,
    shuffle=False
)

x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image_sample')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, img):
        img = functional.relu(self.fc1(img))
        img = functional.relu(self.fc2(img))
        img = functional.softmax(self.fc3(img))
        return img


net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
train_loss = []
for epoch in range(3):
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.view(x.size(0), 28 * 28)
        out = net(x)
        y_oneHot = ont_hot(y)
        loss = functional.mse_loss(out, y_oneHot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

plot_curve(train_loss)

total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), 28 * 28)
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)