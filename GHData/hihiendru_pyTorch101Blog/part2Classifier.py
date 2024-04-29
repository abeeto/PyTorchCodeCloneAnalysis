import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to downsample residual
        # In case the output dimensions of the residual block is not the same
        # as it's input, have a convolutional layer downsample the layer
        # being bought forward by approporate striding and filters
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


def _create_block(in_channels, out_channels, stride):
    return nn.Sequential(
        ResidualBlock(in_channels, out_channels, stride),
        ResidualBlock(out_channels, out_channels, 1)
    )


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)

        # Create blocks
        self.block1 = _create_block(64, 64, stride=1)
        self.block2 = _create_block(64, 128, stride=2)
        self.block3 = _create_block(128, 256, stride=2)
        self.block4 = _create_block(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)

    # A block is just two residual blocks for ResNet18

    def forward(self, x):
        # Output of one layer becomes input to the next
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


data_dir = 'cifar/train'

with open('cifar/labels.txt') as label_file:
    labels = label_file.read().split()
    label_mapping = dict(zip(labels, list(range(len(labels)))))


# Randomly horizontally the image with a probability of 0.5
# Normalise the image with mean and standard deviation of CIFAR dataset
# Reshape it from W  H  C to C  H  W.

def preprocess(image):
    image = np.array(image)

    if random.random() > 0.5:
        image = image[::-1, :, :]

    cifar_mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, -1)
    cifar_std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 1, -1)
    image = (image - cifar_mean) / cifar_std

    image = image.transpose(2, 1, 0)
    return image

# Dataset example
class Cifar10Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_size=0, transforms=None):
        files = os.listdir(data_dir)
        files = [os.path.join(data_dir, x) for x in files]

        if data_size < 0 or data_size > len(files):
            assert ("Data size should be between 0 to number of files in the dataset")

        if data_size == 0:
            data_size = len(files)

        self.data_size = data_size
        self.files = random.sample(files, self.data_size)
        self.transforms = transforms

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image_address = self.files[idx]
        image = Image.open(image_address)
        image = preprocess(image)
        label_name = image_address[:-4].split("_")[-1]
        label = label_mapping[label_name]

        image = image.astype(np.float32)

        if self.transforms:
            image = self.transforms(image)

        return image, label

# Dataloader example (preferred method)

trainset = Cifar10Dataset(data_dir = "cifar/train/", transforms=None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = Cifar10Dataset(data_dir = "cifar/test/", transforms=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

# Train and Evaluate
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     #Check whether a GPU is present.

clf = ResNet()
clf.to(device)   #Put the network on GPU if present

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(clf.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200], gamma=0.1)

for epoch in range(10):
    losses = []
    scheduler.step()
    # Train
    start = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()  # Zero the gradients

        outputs = clf(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Compute the Loss
        loss.backward()  # Compute the Gradients

        optimizer.step()  # Updated the weights
        losses.append(loss.item())
        end = time.time()

        if batch_idx % 100 == 0:
            print('Batch Index : %d Loss : %.3f Time : %.3f seconds ' % (batch_idx, np.mean(losses), end - start))

            start = time.time()
        # Evaluate
    clf.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = clf(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        print('Epoch : %d Test Acc : %.3f' % (epoch, 100. * correct / total))
        print('--------------------------------------------------------------')
    clf.train()
