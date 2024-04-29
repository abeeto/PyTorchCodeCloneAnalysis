"""
Dependencies:
torch: 0.4
torchvision
matplotlib
numpy
"""

#standard library
import os

#third-party library
import torch
import torch.nn as nn
import numpy as numpy
import matplotlib.pyplot as pyplot
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F

torch.manual_seed(1)

#hyper parameters
EPOCH = 10
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False  #MINST is the 1 dim handwrite data set

#Minst dataset download
if not(os.path.exists('./mnist/')) or not(os.listdir('./mnist/')):
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST, 
)

print(train_data.train_data.size())
print(train_data.train_labels.size())

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255
test_y = test_data.test_labels[:2000]
print(test_x)
print(test_y)
# cnn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x
cnn = CNN()
print(cnn)

# train and live test
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH): #one epoch in this case
    for step, (b_x, b_y) in enumerate(train_loader):
        prediction = cnn(b_x)[0]
        loss = loss_func(prediction, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if step % 50 == 0:
            test_out, last_layer = cnn(test_x)
            pred_y = torch.max(test_out, 1)[1].data.numpy()       
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch:', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)




