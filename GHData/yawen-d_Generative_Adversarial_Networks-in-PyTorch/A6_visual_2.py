import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

########
batch_size = 128


# CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Preparing the testing set
transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=1)
        self.conv2 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=2)
        self.conv3 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=1)
        self.conv4 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=2)
        self.conv5 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=1)
        self.conv6 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=1)
        self.conv7 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=1)
        self.conv8 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=(3,3), padding=(1,1), stride=2)
        self.conv1_ln = nn.LayerNorm([196,32,32])
        self.conv2_ln = nn.LayerNorm([196,16,16])
        self.conv3_ln = nn.LayerNorm([196,16,16])
        self.conv4_ln = nn.LayerNorm([196,8,8])
        self.conv5_ln = nn.LayerNorm([196,8,8])
        self.conv6_ln = nn.LayerNorm([196,8,8])
        self.conv7_ln = nn.LayerNorm([196,8,8])
        self.conv8_ln = nn.LayerNorm([196,4,4])
        self.pool = nn.MaxPool2d(4,4)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(in_features=196, out_features=1)
        self.fc10 = nn.Linear(in_features=196, out_features=10)

    def forward(self, x):
        x = F.leaky_relu(self.conv1_ln(self.conv1(x)))
        x = F.leaky_relu(self.conv2_ln(self.conv2(x)))
        x = F.leaky_relu(self.conv3_ln(self.conv3(x)))
        x = F.leaky_relu(self.conv4_ln(self.conv4(x)))
        x = F.leaky_relu(self.conv5_ln(self.conv5(x)))
        x = F.leaky_relu(self.conv6_ln(self.conv6(x)))
        x = F.leaky_relu(self.conv7_ln(self.conv7(x)))
        x = F.leaky_relu(self.conv8_ln(self.conv8(x)))
        x = self.pool(x)
        x = x.view(-1, 196) # reshape x
        out1 = self.fc1(x)
        out2 = self.fc10(x)
        return out1, out2

print('Flag 1')

model1 = Discriminator()
model1.load_state_dict(torch.load('DwoG_pretrained_models/DwoG_model_epoch100.model',map_location='cpu'))
model1.to(device)
model1.eval()

# model2 = Discriminator()
model2 = torch.load('DwG_pretrained_models/discriminator.model',map_location='cpu')
model2.to(device)
model2.eval()


batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch,requires_grad=True).to(device)
Y_batch_alternate = (Y_batch + 1)%10
Y_batch_alternate = Variable(Y_batch_alternate).to(device)
Y_batch = Variable(Y_batch).to(device)


print('Flag 2')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig


X = X_batch.mean(dim=0)
X = X.repeat(10,1,1,1)

Y = torch.arange(10).type(torch.int64)
Y = Variable(Y).to(device)


####### operations with model DwoG


lr = 0.1
weight_decay = 0.001
for i in range(200):
    _, output = model1(X)

    loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).to(device),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples)
plt.savefig('visualization/max_class_DwoG.png', bbox_inches='tight')
plt.close(fig)



####### operations with model DwG

lr = 0.1
weight_decay = 0.001
for i in range(200):
    _, output = model2(X)

    loss = -output[torch.arange(10).type(torch.int64),torch.arange(10).type(torch.int64)]
    gradients = torch.autograd.grad(outputs=loss, inputs=X,
                              grad_outputs=torch.ones(loss.size()).to(device),
                              create_graph=True, retain_graph=False, only_inputs=True)[0]

    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y.data).sum() ) /float(10.0))*100.0
    print(i,accuracy,-loss)

    X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
    X[X>1.0] = 1.0
    X[X<-1.0] = -1.0

## save new images
samples = X.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples)
plt.savefig('visualization/max_class_DwG.png', bbox_inches='tight')
plt.close(fig)



