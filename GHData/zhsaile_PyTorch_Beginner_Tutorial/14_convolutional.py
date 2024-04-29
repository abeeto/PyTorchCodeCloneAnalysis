import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# 0) Hyper parameters
channel = 3
batch_size = 4
lr = 0.001
epochs = 15


# 1) Prepare datasets
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_datasets = torchvision.datasets.CIFAR10(root='./data', train=True,
        transform=transform, download=True)
test_datasets = torchvision.datasets.CIFAR10(root='./data', train=False,
        transform=transform, download=True)

classes = ['plane, car, bird, cat, deer, dog, frog, horse, ship, truck']

train_loader = torch.utils.data.DataLoader(dataset=train_datasets,
        batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
        batch_size=batch_size, shuffle=False, num_workers=2)
train_datas = iter(train_loader)
images, labels = train_datas.next()
print(images.shape, labels.shape)

# 2) Model defined
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,
                kernel_size=(5,5), stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                kernel_size=(5,5), stride=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # first conv layer
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        # second conv layer
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # full connected layer
        out = out.view(-1, 16*5*5)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out


model = Model().to(device)

# 3) loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 4) Training loop
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #update weights
        optimizer.step()

        if (i+1)%1000 == 0:
            print(f'epoch {epoch+1}, step {i+1}/{len(train_loader)},'
                    f'loss = {loss.item():.4f}')

# 5) Test
with torch.no_grad():
    n_corrects = 0
    n_samples = 0
    for images, labels in test_loader:
        n_samples += labels.shape[0]
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        
        n_corrects += (predictions == labels).sum().item()

    accuracy = 100 * n_corrects / n_samples
    print(f'accuracy: {accuracy:.4f}')

