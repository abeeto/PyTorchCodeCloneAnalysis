import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, batch_size=16, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=False, batch_size=16, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device('cuda')


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


net = Net()
net = net.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)


def learn(data, target):
    optimizer.zero_grad()
    pred = net(data)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()
    return loss.data.item()


def run(total=3):
    import time
    start = time.time()
    for epoch in range(total):
        for i, (data, target) in enumerate(train_loader):
            loss = learn(data.to(device), target.to(device))
            print(f"loss : {loss}", end="\r")
    print()
    return time.time() - start


def test():
    total = 0
    correct = 0
    for data, target in test_loader:
        _, predict = torch.max(net(data.to(device)).data, 1)
        total += target.size(0)
        correct += (predict == target.to(device)).sum().item()
    return correct / total
