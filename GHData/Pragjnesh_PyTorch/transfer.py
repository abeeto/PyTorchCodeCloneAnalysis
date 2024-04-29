from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from caltech256 import Caltech256
from torchvision import models,transforms

## Load Data
#TODO: Data augmentation for training
transform = transforms.Compose(
    [
        transforms.Scale((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)
data_dir = '/datasets/Caltech256/256_ObjectCategories/'
caltech256_train = Caltech256(data_dir, transform, train=True)
caltech256_test = Caltech256(data_dir, transform, train=False)
# Credit to https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb for sampler example
num_train = len(caltech256_train)
indices = list(range(num_train))
split = int(np.floor(0.1 * num_train))
np.random.shuffle(indices)
train_idx, val_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

train_data = DataLoader(
    dataset = caltech256_train,
    batch_size = 32,
    sampler = train_sampler,
    shuffle = False,
    num_workers = 4
)

val_data = DataLoader(
    dataset = caltech256_train,
    batch_size = 32,
    sampler = val_sampler,
    shuffle = False,
    num_workers = 4
)

test_data = DataLoader(
    dataset = caltech256_test,
    batch_size = 8,
    shuffle = True,
    num_workers = 4
)

dataloaders = {}
dataloaders['train'] = train_data
dataloaders['val'] = val_data
dataloaders['test'] = test_data
use_gpu = torch.cuda.is_available()

## Train the neural network
# Credit to http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html for framework

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val','test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs,labels.view(-1).long())

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_conv = models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_featuresm
model_conv.fc = nn.Linear(num_ftrs, 256)

if use_gpu:
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opoosed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
