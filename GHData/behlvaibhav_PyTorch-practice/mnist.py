from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
from torch import optim
import numpy as np

_tasks = transforms.Compose((
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
))
# Load the MNIST dataset and apply transformation
mnist = MNIST("data", download=True, train=True, transform=_tasks)

# Creating training and validation split
split = int(0.8 * len(mnist))
index_list = list(range(len(mnist)))
train_idx, valid_idx = index_list[:split], index_list[split:]

# Creating sampler objects using SubsetRandomSampler
tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)

# create iterator objects for train and valid datasets
train_loader = DataLoader(mnist, batch_size=256, sampler=tr_sampler)
valid_loader = DataLoader(mnist, batch_size=256, sampler=val_sampler)


# MLP
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 128)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.hidden(x)
        x = torch.sigmoid(x)
        x = self.output(x)
        return x


model = Model()

# Define loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01,
                      weight_decay=1e-6, momentum=0.9,
                      nesterov=True)
for epoch in range(1, 31):
    train_loss, valid_loss = [], []
    # Training part
    model.train()
    for data, target in train_loader:
        # zero the parameters gradients
        optimizer.zero_grad()
        # forward propagation
        output = model(data)
        # loss calculation
        loss = loss_function(output, target)
        # backward propagation
        loss.backward()
        # weight optimization
        optimizer.step()
        train_loss.append(loss.item())
    # model evaluation
    model.eval()
    for data, target in valid_loader:
        output = model(data)
        loss = loss_function(output, target)
        valid_loss.append(loss.item())
    print("Epoch: ", epoch, "Training Loss: ", np.mean(train_loss), "Validation Loss: ", np.mean(valid_loss))

# Now we'll make predictions on validation data
data_iter = iter(valid_loader)
data, labels = data_iter.next()
output = model(data)
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy())
print ("Actual:", labels[:10])
print ("Predicted:", preds[:10])
