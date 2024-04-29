import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import matplotlib as plt
from matplotlib.pyplot import *
from matplotlib import pyplot as plt

NUM_TRAIN = 49000
cifar_dict = ['airplane', 'automobile', 'bird', 'cat', 'deer','dog','frog','horse','ship','truck' ]
# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914), (0.2023))
            ])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cifar10_train = dset.CIFAR10(
    './cs231n/datasets',
    train=True, download=True,
    transform=transform
)
loader_train = DataLoader(
    cifar10_train,
    batch_size=64,
    sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN))
)
cifar10_val = dset.CIFAR10(
    './cs231n/datasets',
    train=True, download=True,
    transform=transform
)
loader_val = DataLoader(
    cifar10_val, batch_size=64, 
    sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000))
)

cifar10_test = dset.CIFAR10(
    './cs231n/datasets',
    train=False, download=True,
    transform=transform
)
loader_test = DataLoader(cifar10_test, batch_size=64)

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

def predict(loader, model):
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            grid = np.concatenate([np.swapaxes(np.swapaxes(x[i].cpu().reshape(3,32,32),0,1),1,2) for i in range(10)], axis = 1)
            imshow(grid)
            final_pred = torch.argmax(scores,axis = 1)[:10].cpu()
            for i in final_pred:
              print(cifar_dict[i], end = " ")
            print()
            plt.show()

def train(model, optimizer, loader_train, loader_val, print_every = 100,epochs=1):
    model = model.to(device=device)
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype = dtype)
            y = y.to(device=device, dtype = torch.long)
            scores = model(x)
            loss = F.cross_entropy(scores,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if t % print_every == 0:
                print('Iteration %d, loss = %.4f'%(t,loss.item()))
                check(loader_val,model)
                print()
                
def check(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct+= (preds==y).sum()
            num_samples+= preds.size(0)
        acc = float(num_correct)/num_samples
        print('Got %d / %d correct (%.2f)'%(num_correct, num_samples, 100*acc))