import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.models as pre_def_models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import math
import ssl

is_gpu_available = False
if torch.cuda.is_available():
    is_gpu_available = True

if not is_gpu_available:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

data_path = 'data/train/'

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

dataset = datasets.ImageFolder(
        root=data_path,
        transform=transform)
num_classes = len(dataset.classes)

validation_split = 0.2
shuffle_dataset = True
random_seed= 42
# obtain training indices that will be used for validation
num_dataset = len(dataset)
# percentage of training set to use as validation
valid_size = int(num_dataset * validation_split)

indices = list(range(num_dataset))
split = int(np.floor(validation_split * num_dataset))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]


# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

batch_size, num_workers = 32, 4

# prepare data loaders (combine dataset and sampler)
train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)

# download pre-trained model
ssl._create_default_https_context = ssl._create_unverified_context
model = pre_def_models.resnet18(pretrained=True)

## freezing the training for all layers
#for param in model.parameters():
#    param.require_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# move tensors to GPU if CUDA is available
if is_gpu_available:
    model.cuda()

import torch.optim as optim
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
# specify optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)


def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data, labels) in enumerate(data_loader):
        if batch_idx % 10 == 0:
            print('Phase = ', phase, ' Epoch = ', epoch, ' Batch idx = ', batch_idx)
        if is_gpu_available:
            data, labels = data.cuda(), labels.cuda()
        data, labels = Variable(data, volatile), Variable(labels)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        #updating running loss
        running_loss += loss.item() * data.size(0)
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(labels.data.view_as(preds)).cpu().sum().item()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.sampler.indices)
    accuracy = 100. * running_correct / len(data_loader.sampler.indices)

    #print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy

# number of epochs to train the model
n_epochs = 20
# track change in loss
val_epoch_loss_min = np.Inf
train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []

print('Training Start...')
for epoch in range(1, n_epochs):
    train_epoch_loss, train_epoch_accuracy = fit(epoch, model, train_data_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, model, valid_data_loader, phase='validation')
    train_losses.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
    ### saving model if validation model is less than previous validation loss
    if val_epoch_loss < val_epoch_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  ...'.format(val_epoch_loss_min, val_epoch_loss))
        print('Saving Model...')
        torch.save(model.state_dict(), 'model_cat_vs_dog_pre_train_resnet18.pt')
        val_epoch_loss_min = val_epoch_loss

import csv
f = open('plot_values.csv', 'w')
writer = csv.writer(f, delimiter=',')
writer.writerow(train_losses)
writer.writerow(train_accuracy)
writer.writerow(val_losses)
writer.writerow(val_accuracy)

plt.plot(range(1, len(train_losses)+1), train_losses, 'bo', label = 'training loss')
plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label = 'validation loss')
plt.legend()
plt.savefig('train_validation_loss.png')
plt.gcf().clear()

plt.plot(range(1, len(train_accuracy)+1), train_accuracy, 'bo', label = 'train accuracy')
plt.plot(range(1, len(val_accuracy)+1), val_accuracy, 'r', label = 'val accuracy')
plt.legend()
plt.savefig('train_validation_accuracy.png')
plt.gcf().clear()

print('Exit Code.... ')