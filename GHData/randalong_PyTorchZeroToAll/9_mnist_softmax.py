import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

batch_size = 64

train_loader = DataLoader(datasets.MNIST('./data', train = True, download = False, transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size = batch_size, shuffle = True)

test_loader = DataLoader(datasets.MNIST('./data', train = False, download = False, transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size = batch_size, shuffle = True)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(784, 520)
        self.l2 = torch.nn.Linear(520, 320)
        self.l3 = torch.nn.Linear(320, 240)
        self.l4 = torch.nn.Linear(240, 120)
        self.l5 = torch.nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, x.size(0))
        out1 = F.relu(self.l1(x))  # out1: 784-520
        out2 = F.relu(self.l2(out1))  # out2: 520-320
        out3 = F.relu(self.l3(out2))  # out3: 320-240
        out4 = F.relu(self.l4(out3))  # out4: 240-120
        y_pred = self.l5(out4)  # y_pred: 120-10
        return y_pred


model = Model()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)

