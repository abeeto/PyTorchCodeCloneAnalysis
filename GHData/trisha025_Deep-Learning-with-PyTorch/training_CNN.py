import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True) 

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size =5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features = 60, out_features = 10)

    def forward(self, t):
        # (1) Input layer
        t = t

        # (2) Conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) Conv Layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) Linear Layer
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) Linear Layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) out layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)

        return t

#training set
train_set = torchvision.datasets.FashionMNIST(
    root= './data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

#training with a single batch
network = Network()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
batch = next(iter(train_loader))
images, labels = batch

    #calculating the loss
preds = network(images)
loss = F.cross_entropy(preds, labels)
print(loss.item())

    #calculating the gradient
print(network.conv1.weight.grad)

loss.backward()
print(network.conv1.weight.grad.shape)

    #update weights
optimizer = optim.Adam(network.parameters(), lr=0.01)
print('loss 1: ',loss.item())
#print(get_num_correct(preds, labels))

optimizer.step()

preds = network(images)
loss = F.cross_entropy(preds, labels)

print('loss2: ',loss.item())
#print(get_num_correct(preds, labels))

