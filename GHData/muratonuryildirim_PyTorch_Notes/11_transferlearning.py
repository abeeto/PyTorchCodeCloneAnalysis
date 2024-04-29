import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch_functions import Identity, check_accuracy

'''
0. prepare dataset
1. design model (input, output, forward pass)
2. initiate the model
3. define loss and optimizer
4. train the model (loss)
    - forward pass: compute prediction and loss
    - backward pass: update weights
5. test the model (accuracy)
'''

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 0. prepare dataset
batch_size = 1024
train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# 1. design model
in_channels = 3
num_classes = 10
learning_rate = 0.001

#   - load pre-trained network and freeze its parameters
#     (after the freezing only modified layers will be updated)
model = torchvision.models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# modify the model
print(model)  # let's say we want to remove (a)avgpool and also change (b)the classifier
# (a) avgpool
model.avgpool = Identity()

# (b) the classifier
# i. Changing the whole classifier
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 10)).to(device)

# ii. Changing a part of the classifier (don't forget to make rest of the classifier Identity)
# model.classifier[0] = nn.Linear(512,10).to(device)


# 3. define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. train the model (loss)
num_epochs = 1
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 5. test the model (accuracy)
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
