import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

# Number of sub processes
num_workers = 0

# NUmber of samples per batch
batch_size = 20
valid_size = 0.2

transform = transforms.ToTensor()

# Train and Test
train_data = datasets.MNIST("MNIST_data", download=False, train=True, transform=transform)
test_data = datasets.MNIST("MNIST_data", download=False, train=False, transform=transform)


# Validation Split
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        hidden_1 = 512
        hidden_2 = 512
        self.fc1 = nn.Linear(28*28, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Flatten
        x = x.view(-1, 28*28)
        # Activation function on hidden layer
        x = F.relu(self.fc1(x))
        # Dropout
        x = self.dropout(x)
        # Activation function on hidden layer
        x = F.relu(self.fc2(x))
        # Dropout
        x = self.dropout(x)
        # Output Layer
        x = self.fc3(x)
        return x


model = Net()
# print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training the network
n_epochs = 50
valid_loss_min = np.Inf

for epoch in range(n_epochs):
    train_loss = 0.0
    valid_loss = 0.0
    train_acc = 0.0
    valid_acc = 0.0

    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = torch.max(output, 1)
        train_acc += torch.mean(pred.eq(labels).type(torch.FloatTensor))

    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            output = model(images)
            loss = criterion(output, labels)
            valid_loss += loss.item()
            _, pred = torch.max(output, 1)
            valid_acc += torch.mean(pred.eq(labels).type(torch.FloatTensor))

    print('Epoch: {}/{} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.2f}\t'
          'Validation Loss: {:.6f} \tValidation Accuracy: {:.2f}'.format(
            epoch+1, n_epochs,
            train_loss/len(train_loader),
            train_acc/len(train_loader),
            valid_loss/len(valid_loader),
            valid_acc/len(valid_loader)))

    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = valid_loss


# model with lowest loss
model.load_state_dict(torch.load('model.pt'))


test_loss = 0.0
test_acc = 0.0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        loss = criterion(output, labels)
        test_loss += loss.item() * images.size(0)
        _, pred = torch.max(output, 1)
        test_acc += torch.mean(pred.eq(labels).type(torch.FloatTensor))

test_loss = test_loss/len(test_loader)
test_acc = test_acc/len(test_loader)
print('Test Loss: {:.6f} \tTest Accuracy: {:.2f}'.format(
    test_loss,
    test_acc))

# JIT Save
jit_model = torch.jit.script(model)
jit_model.save('ann_model_jit.pt')
