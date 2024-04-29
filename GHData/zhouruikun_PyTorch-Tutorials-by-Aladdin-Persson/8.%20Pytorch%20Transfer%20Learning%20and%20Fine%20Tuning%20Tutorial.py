# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary
from torchvision.models import VGG16_Weights
# %%
# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# hyper parameters
in_channels = 3
num_classes = 100
learning_rate = 3e-5
batch_size = 128
num_epochs = 5


class Identify(nn.Module):
    def __init__(self):
        super(Identify, self).__init__()

    def forward(self, x):
        return x
# %% check accuracy on training & test to see how good our model


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("checking accuracy on training data")
    else:
        print("checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            score = model(x)
            _, prediction = score.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()
# %%


model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
# for param in model.parameters():
#     param.requires_grad = False
# model.avgpool = Identify()
# model.classifier = nn.Linear(2048, num_classes)
model.to(device)
summary(model, (3, 64, 64))
# %%  load data
train_dataset = datasets.CIFAR100(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.CIFAR100(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
# %% Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# %% train

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        targets = targets.to(device)
        data = data.to(device)
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'epoch {epoch}: {batch_size*batch_idx}/{len(train_dataset.data)} with loss = {loss.item()}')
    check_accuracy(test_loader, model)
    check_accuracy(train_loader, model)
# %%


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
# %%
print(device)
