import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=120)

#download data
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    
    ])
)

#data loading into an object
#train_loader = torch.utils.data.DataLoader(
#    train_set, batch_size=100)

print("Total number of traning images: ", len(train_set))

print("Training labels: ", train_set.targets)

print("Number of images in each class: ", train_set.targets.bincount())

#to access an individual sample
#sample = next(iter(train_set))
#print(len(sample)) # =2 (i) image (ii) label

#print(type(sample))
#image, label = sample

#sample shape
#print(image.shape)

#plt.imshow(image.squeeze(), cmap = 'gray')
#print('label: ', label)
#plt.show()

#to access a batch
#batch = next(iter(train_loader))

#print(len(batch))
#print(type(batch))
#images, labels = batch

#batch shape
#print(images.shape)
#print(labels.shape)

#batch visualisation
#grid = torchvision.utils.make_grid(images, nrow=10)
#plt.figure(figsize=(15,15))
#plt.imshow(np.transpose(grid, (1,2,0)))
#print('labels: ', labels)
#plt.show()

#NN Class
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features = 60, out_features=10)
            
    def forward(self, t):
        # (1) input layer
        t=t

        #(2) Conv layer
        t= self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) Conv Layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size= 2, stride=2)

        # (4) Linear layer
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) Linear Layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)

        return t

#torch.set_grad_enabled(False)

network = Network()
#print(network)

data_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=10
)

#batch = next(iter(data_loader))
#images, labels = batch

#print(image.shape)
#print(labels.shape)

#preds = network(images)
#print(preds.shape)
#print(preds)
#print(preds.argmax(dim=1))
#print(labels)
#print(preds.argmax(dim=1).eq(labels))
#print(preds.argmax(dim=1).eq(labels).sum())

sample = next(iter(train_set))
image, label = sample

output = network(image.unsqueeze(0))
print(output)

