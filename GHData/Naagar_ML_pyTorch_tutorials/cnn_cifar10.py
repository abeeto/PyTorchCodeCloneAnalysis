#  cnn_cifar10

# only for images 
# or videos 
# activation layesr, pooling layers one or more fulley connected class 

# output woth summing all the values 
# same thing same filter operation then se slide the filter , padding and stride in practivate 
# Poolong layers downsampling 2x2 used to redused the computational cost also helps in over fitting 
# # dear sir i want to designe a network for smaller data set in the CNN model
# 
# 
#  
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms
import torchvision 
import matplotlib.pyplot as plt 
import numpy 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters 

num_epoch = 4
batch_size = 4
lr_rate = 0.001

# dataset has PIL images of rnage [0, 1]
# we transform them to Tensor of normalized rnage [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                            download=True, transform=transform)
                                        
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'cat', 'bird', 'dog', 'deer', 'frog', 'horse', 'ship', 'truck')


# impliment conv net

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)       ##out  [4, 3, 32, 32]
        self.pool = nn.MaxPool2d(2, 2)      ##  [4, 6, 28, 28]
        self.conv2 = nn.Conv2d(6,16,5)       #  [4, 16, 10, 10 ]
        self.fc1 = nn.Linear(16*5*5, 110)   #   4, 110
        self.fc2 = nn.Linear(110, 84)       #      4, 84
        self.fc3 = nn.Linear(84, 10)        #      4, 10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # first conv with pool and relu(which does not change the size )
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)                  # flattnig the 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

model = CNN().to(device)

criterion = nn. CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr_rate)

n_total_step = len(train_loader)

for epoch in range(numpy):
    for i, (images,labels) in enumerate(train_loader):
        # original shape = [4, 3, 32, 32 ] = 4, 3, 1024
        # imput_layers: 3 imput_channels, 6 output channels, 5kernal size

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        output = model(images)

        loss = criterion(output, labels)

        # Backward and optimizer  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # setting zero 

        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epoch}], Step [{i+1}/{n_total_step}], loss : {loss.item():.4f}')
    print('Finished traning ')

with torch.no_grad():
    n_correct = 0
    n_sample = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # max return(value, index)
        _, predicted = torch.max(outputs,1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        for i in range(batch_size):
            label = label[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1    
    acc = 100 * n_correct / n_sample
    print(f'Accuracy of Network:, {acc} %')

for i in range(10):
    acc = 100.0 *n_class_correct[i] / n_class_samples[i]
    print(f'Accuracy of {classes[i]}, {acc} %') 