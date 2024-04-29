import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.models as pre_def_models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

import math
import ssl

# check if CUDA is available
is_gpu_available = torch.cuda.is_available()

if not is_gpu_available:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

data_path = 'data/CatVsDog/train/'

def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

train_data = datasets.ImageFolder(
        root=data_path,
        transform=transform)
num_classes = len(train_data.classes)

# obtain training indices that will be used for validation
num_train = len(train_data)
# percentage of training set to use as validation
valid_size = int(num_train * 0.2)

indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size))
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

batch_size, num_workers = 32, 4
# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)

# download pre-trained model
ssl._create_default_https_context = ssl._create_unverified_context
model = pre_def_models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# number of epochs to train the model
n_epochs = 30
epoch_save_strt = math.ceil(n_epochs * 0.1)
valid_loss_min = np.Inf # track change in validation loss

import torch.optim as optim
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# move tensors to GPU if CUDA is available
if is_gpu_available:
    model.cuda()

print('Training Start...')
for epoch in range(1, n_epochs+1):
    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    # train the model set the model in train model
    model.train()
    batch_counter = 1
    for data, labels in train_loader:
        batch_counter += 1
        if batch_counter % 10 == 0:
            print('Process Going On Batch Number = ', batch_counter)
        # move tensors to GPU if CUDA is available
        if is_gpu_available:
            data, labels = data.cuda(), labels.cuda()

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, labels)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item() * data.size(0)

    # validate the model #
    model.eval()
    for val_data, labels in valid_loader:
        # move tensors to GPU if CUDA is available
        if is_gpu_available:
            val_data, labels = val_data.cuda(), labels.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(val_data)
        # calculate the batch loss
        loss = criterion(output, labels)
        # update average validation loss
        valid_loss += loss.item() * data.size(0)

    # calculate average losses
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  ...'.format(
            valid_loss_min,
            valid_loss))
        if epoch > epoch_save_strt:
            print('Saving Model...')
            torch.save(model.state_dict(), 'model_cat_vs_dog_pre_train_resnet18.pt')
        valid_loss_min = valid_loss

print('Training finish  .. exiting setup')