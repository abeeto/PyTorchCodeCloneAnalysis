import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

num_workers = 0
batch_size = 20
valid_size = 0.2

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_data = datasets.CIFAR10("CIFAR_data", train=True, download=False, transform=transform)
test_data = datasets.CIFAR10("CIFAR_data", train=False, download=False, transform=transform)

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


# CNN Architecture
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 10
valid_loss_min = np.Inf

for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    train_acc = 0.0
    valid_loss = 0.0
    valid_acc = 0.0

    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        train_loss += loss
        _, pred = torch.max(output, 1)
        train_acc += torch.mean(pred.eq(labels).type(torch.FloatTensor))

    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            output = model(images)
            loss = criterion(output, labels)
            valid_loss += loss
            _, pred = torch.max(output, 1)
            valid_acc += torch.mean(pred.eq(labels).type(torch.FloatTensor))

    print('Epoch: {}/{} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.2f}\t'
          'Validation Loss: {:.6f} \tValidation Accuracy: {:.2f}'.format(
            epoch, n_epochs,
            train_loss / len(train_loader),
            train_acc / len(train_loader),
            valid_loss / len(valid_loader),
            valid_acc / len(valid_loader)))

    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'cnn_model.pt')
        valid_loss_min = valid_loss


# model with lowest loss
model.load_state_dict(torch.load('cnn_model.pt'))

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
jit_model.save('cnn_model_jit.pt')
