import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt

DATA_SET_SIZE = 60000
DATA_SET_TEST = 60000 * 0.8
DATA_SET_VALIDATION = 60000 * 0.2
EPOCHS=10

transforms = transforms.Compose([
    transforms.ToTensor()  # ,
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
indices = list(range(DATA_SET_SIZE))
train_indices, val_indices = indices[:int(DATA_SET_SIZE * 0.8)], indices[int(DATA_SET_SIZE * 0.8):]
train_sam, validation_sam = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)
train_set = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=True, download=True,
                          transform=transforms),
    batch_size=64, sampler=train_sam)
validation_set = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./data', train=True, download=True,
                          transform=transforms),
    batch_size=64, sampler=validation_sam)


def train(model, optimizer):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_set):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    test_loss_data=[]
    test_accurency_data=[]
    for data, target in validation_set:
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).cpu().sum()
    test_loss /= int(DATA_SET_VALIDATION)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, int(DATA_SET_VALIDATION),
        100. * correct / int(DATA_SET_VALIDATION)))
    test_loss.append("{:.4f}",test_loss)



def plot(plt_title,xlable,ylable,save_Fige_file,epochs,data):
    plt.plot(range(epochs), data, 'bo-', color='g')
    plt.title("Perceptron")
    # plt.locator_params(axis='x', nbins=200)
    plt.xlabel("Ephochs")
    plt.ylabel("Success rate")
    plt.savefig("Perceptron", bbox_inches="tight")
    plt.close()


class FirstNet(nn.Module):
    def __init__(self, image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = FirstNet(image_size=28 * 28)
optimizer = optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, int(EPOCHS) + 1):
    train(model, optimizer)
    test(model)
