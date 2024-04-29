import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
import matplotlib.pyplot as plt

class ChunkSampler(sampler.Sampler):
    def __init__(self,num_samples,start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start,self.start+self.num_samples))

    def __len__(self):
        return self.num_samples

#total training samples = 60000
#size of samples is 1x28x28
NUM_TRAIN = 58000
NUM_VAL = 2000

batch_size = 64

mnist_train = dset.MNIST('./data',train=True,download=True, transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=batch_size, sampler=ChunkSampler(NUM_TRAIN,0))

mnist_val = dset.MNIST('./data',train=True,download=True, transform=T.ToTensor())
loader_val = DataLoader(mnist_val, batch_size=batch_size, sampler=ChunkSampler(NUM_VAL,NUM_TRAIN))

mnist_test = dset.MNIST('./data',train=False,download=True, transform=T.ToTensor())
loader_test = DataLoader(mnist_test, batch_size=batch_size)

dtype = torch.FloatTensor #CPU datatype

print_every = 100

def reset(m):
    if hasattr(m,'reset_parameters'):
        m.reset_parameters()

class Flatten(nn.Module):
    def forward(self,x):
        N,C,H,W = x.size()
        return x.view(N,-1)


def train(model,loss_fn,optimizer,num_epochs):
    for epoch in range(num_epochs):
        acc_hist.append(check_accuracy(model,loader_val))
        print('Starting epoch %d/%d' %(epoch+1,num_epochs))
        model.train()
        for t, (x,y) in enumerate(loader_train):
            x_var = Variable(x.type(dtype))
            y_var = Variable(y.type(dtype).long())
             
            scores = model(x_var)

            loss = loss_fn(scores,y_var)
            loss_hist.append(loss.data[0])
            if((t+1)%print_every == 0):
                print('%d, loss = %.4f' % (t+1,loss.data[0]))

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

def check_accuracy(model,loader):
    if loader.dataset.train:
        print("Checking accuracy on validation set...")
    else:
        print("Cheching accuracy on test set...")
    num_correct = 0
    num_samples = 0
    model.eval() # put model into evaluation mode
    for x,y in loader:
        x_var = Variable(x.type(dtype), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds==y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct)/num_samples
    print("Got %d/%d correct: %.2f" %(num_correct,num_samples,acc*100))
    return acc


simple_nn = nn.Sequential(
        Flatten(),
        nn.Linear(784,500),
        nn.ReLU(),
        nn.BatchNorm1d(500),
        nn.Linear(500,784),
        nn.ReLU(),
        nn.BatchNorm1d(784),
        nn.Linear(784,10)
        )
loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.Adam(simple_nn.parameters(),lr=1e-2)


loss_hist = []
acc_hist = []
simple_nn.apply(reset)
train(simple_nn,loss_fn,optimizer,num_epochs = 10)
check_accuracy(simple_nn,loader_val)

plt.plot(loss_hist,'b-',acc_hist,'r-')
plt.show()
