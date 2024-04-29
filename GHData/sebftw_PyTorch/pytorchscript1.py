import torch, torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torch.optim as optim

class ELUTest(nn.Module):
    def __init__(self):
        
        #super(ELUTest, self).__init__()

conv = torch.nn.Sequential(
    nn.Conv2d(1, 20, 5, 1),
    nn.ELU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(20, 50, 5, 1),
    nn.ELU(),
    nn.MaxPool2d(2, 2)
)

fullc = torch.nn.Sequential(
    nn.Linear(4*4*50, 500),
    nn.ReLU(),
    nn.Linear(500, 10))

softmax = nn.LogSoftmax(1)

conv.load_state_dict(torch.load('pytorchtestnet-conv.pt'))
fullc.load_state_dict(torch.load('pytorchtestnet-fullc.pt'))



torch.manual_seed(7)
transform = transforms.ToTensor()
trainset = torchvision.datasets.MNIST('mnist', download = True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle=True, num_workers=0)



ysource = 1 #torch.ones(1, dtype=torch.long).mul(0) #get images of 0
xvals = [x for i, (x, y) in enumerate(trainloader, 0) if y[0].item() == ysource] #Python magic
xfullset = torch.stack(xvals, dim = 0).view(len(xvals), 1, 28, 28)
ytarget = 1 #transform to 1

xvals = xvals[1:3]
batchloader = data.BatchSampler(data.RandomSampler(xfullset), batch_size=64*2, drop_last=True)


#x, y = next(iter(trainloader))
#y[0] = 1


#x = torch.add(x, 0.05)#torch.mul(torch.randn(x.size()), 0.005)
#x = x.clamp(0, 1)
#x = torch.add(x, -0.05)
#x = torch.add(x, )
#x = x.clamp(0, 1)

#img = x.detach()
#img = img.view([1] + list(img.size()[2:4]))
#img = transforms.ToPILImage()(img)
#img.show()

rando = torch.tensor(torch.randn(1, 1, 28, 28).mul(0.001), requires_grad=True)

loss_fn = torch.nn.CrossEntropyLoss()


lr = 1e-1 #3e-4
lambd = 0
optimizer = optim.Adagrad([rando], lr=lr)
for  i, x in enumerate(batchloader, 0):
    y_pred = xfullset[x]
    optimizer.zero_grad()
    y_pred = torch.add(y_pred, rando).clamp(0, 1)
    y_pred = conv(y_pred)
    y_pred = fullc(y_pred.view(-1, 4 * 4 * 50))
    #y_pred = softmax(y_pred)
    #y_pred = torch.mm(y_pred, coord)
    loss = loss_fn(y_pred, torch.ones(len(x), dtype=torch.long).mul(ytarget))
    loss = torch.add(loss, rando.abs().sum(3).sum(2).mul(lambd/(2*rando.numel()))) #pow(2)
    print(i, loss.item())

    loss.backward()
    optimizer.step()

img = torch.add(xfullset[0], rando.detach()).clamp(0, 1)
y_pred = conv(torch.add(xfullset, rando.detach()).clamp(0, 1))
y_pred = fullc(y_pred.view(-1, 4 * 4 * 50))
y_pred = torch.argmax(y_pred, 1)

correct = ( y_pred == ytarget )
correct = torch.sum(correct.int())
correct = 1 - correct.item() / y_pred.size(0)

print(round(correct * 100 * 100)/100)

img = img.view([1] + list(img.size()[2:4]))
img = transforms.ToPILImage()(img)
img.show()