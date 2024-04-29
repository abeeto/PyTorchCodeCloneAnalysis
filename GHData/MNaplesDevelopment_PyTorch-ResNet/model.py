import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets


learning_rate = 1e-3
batch_size = 64
num_classes = 10
color = True

num_channels = 3 if color else 1

trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()])

train_data = datasets.CIFAR10(root='/data/', train=True, transform=trans, download=True)
test_data = datasets.CIFAR10(root='/data/', train=False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.res1 = ResidualBlock(16, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.res2 = ResidualBlock(32, 32) # Max pool and res block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.res3 = ResidualBlock(64, 64) # Max pool and res block
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.res4 = ResidualBlock(128, 128)
        self.fc = nn.Linear(128*4*4, num_classes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.mp2 = nn.MaxPool2d(2)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.res2(x)
        x = self.mp2(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.res3(x)
        x = self.mp2(x)
        x = self.res3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.res4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



model = ResNet().cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_batches = len(train_loader)


def train(epochs):
    for epoch in range(epochs):
        for i, (batch, labels) in enumerate(train_loader):
            batch, labels = batch.cuda(), labels.cuda()
            model.zero_grad()

            output = model(batch)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f'Epoch: {epoch+1}/{epochs}\tBatch: {i}/{num_batches}\tLoss: {loss}')

train(10)

correct = 0
total = 0
with torch.no_grad():
  for images, labels in test_loader:
      images = torch.tensor(images).cuda()
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted.cpu() == labels).sum()

print('Accuracy of the model on the test images: %d %%' % (100 * correct / total))
