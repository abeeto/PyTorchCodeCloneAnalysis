import sys
# from nets import *
import argparse
import multiprocessing
import torchvision
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
# importing the libraries
import torchvision.transforms.functional as F
from colorama import init
from efficientnet_pytorch import EfficientNet
# PyTorch libraries and modules
from loguru import logger
from termcolor import colored
from torch.optim import lr_scheduler
from torchsummary import summary
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
init()
# Construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, help="Training mode: ResNet18/EfficientNet")
args = vars(ap.parse_args())

# Set training mode
train_mode = args["mode"]

# Set the train and validation directory paths
train_directory = '/data/stars/user/eboughos/sex/all/combined/train'

valid_directory = '/data/stars/user/eboughos/sex/all/combined/val'

# Set the model save path
if train_mode == 'ResNet18':
    PATH = "./ResNet18.pth"
elif train_mode == 'EfficientNet':
    PATH = "./EfficientNet.pth"

# Batch size
bs = 32

# Number of epochs
num_epochs = 100

# Number of classes
num_classes = 2

# Number of workers
num_cpu = multiprocessing.cpu_count()

if train_mode == 'ResNet18':
    fid = open('ResNet.log', 'w')

elif train_mode == 'EfficientNet':
    fid = open('EfficientNet.log', 'w')

logger.add(
    sink=fid,
    colorize=True
)

# padding function
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')


# Applying transforms to the data
image_transforms = {
    'train': transforms.Compose([
        SquarePad(),
        transforms.Resize((128, 128)),
        # transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=0.5)]), p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        SquarePad(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Load data from folders
dataset = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}

# Size of train and validation data
dataset_sizes = {
    'train': len(dataset['train']),
    'valid': len(dataset['valid'])
}

# Create iterators for data loading
dataloaders = {
    'train': data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                             num_workers=num_cpu, pin_memory=True, drop_last=True),
    'valid': data.DataLoader(dataset['valid'], batch_size=bs, shuffle=True,
                             num_workers=num_cpu, pin_memory=True, drop_last=True)
}

# # Load data from folders
# dataset = {
#     'train': torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=image_transforms['train']),
#     'valid': torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=image_transforms['valid'])
# }
#
# # Size of train and validation data
# dataset_sizes = {
#     'train': len(dataset['train']),
#     'valid': len(dataset['valid'])
# }
#
# # Create iterators for data loading
# dataloaders = {
#     'train': data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
#                              num_workers=num_cpu, pin_memory=True, drop_last=True),
#     'valid': data.DataLoader(dataset['valid'], batch_size=bs, shuffle=True,
#                              num_workers=num_cpu, pin_memory=True, drop_last=True)
# }

# # Class names or target labels
# class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# print("Classes:", class_names)
# logger.info('Classes are : {}'.format(','.join(class_names)))

# Class names or target labels
class_names = dataset['train'].classes
print("Classes:", class_names)
logger.info('Classes are : {}'.format(','.join(class_names)))

# Print the train and validation data sizes
logger.info('Training set size : {}.'.format(dataset_sizes['train']))
logger.info('Validation set size : {}.'.format(dataset_sizes['valid']))

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#MODEL Mode
if train_mode == 'ResNet18':
    num_epochs = 100

    # Load a pretrained model - Resnet18
    logger.info("Loading resnet18 for transfer learning ...")
    model_ft = models.resnet18(pretrained=True)

    # # Modify fc layers to match num_classes
    num_ftrs = model_ft.fc.in_features

   #Adding layers
    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )

elif train_mode == 'EfficientNet':

    # Load a pretrained model - MobilenetV2
    logger.info("Loading EfficientNet for transfer learning ...")
    model_ft = EfficientNet.from_pretrained('efficientnet-b7')

    # Modify fc layers to match num_classes
    num_ftrs = model_ft._fc.in_features

    #Adding Layers
    model_ft._fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=128),
        nn.Linear(128, 2),
    )

# Transfer the model to GPU
model_ft = model_ft.to(device)

# Print model summary
logger.info('Model Summary')
logger.info(summary(model_ft, input_size=(3, 128, 128)))

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.00001)
#optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01, weight_decay=0.000001)

# Learning rate decay
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.5)


def set_model_mode(model, mode):
    if mode == 'train':
        model.train()

        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        if train_mode == "ResNet18":
            for m in model.fc.modules():
                m.train()
        if train_mode == "EfficientNet":
            for m in model._fc.modules():
                m.train()
    else:
        model.eval()
    return model


def train_model(model, criterion, optimizer, scheduler, num_epochs=100):
    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            model = set_model_mode(model, phase)
            running_loss = 0.0
            running_corrects = 0
            iter = 1

            # Iterate over data.
            running_size = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                # with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                outputs = nn.Softmax()(outputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_size += inputs.size(0)
                running_loss += loss.item()
                current_correct = torch.sum(preds == labels.data).double()
                running_corrects += current_correct


                if iter % 100 == 0:
                    if phase == 'valid':
                        logger.info(colored('{} : Epoch ( {}/{} ), Iteration {} : iter_loss : {:.3f}\t running_loss : '
                                            '{:.3f}\t iter_acc : {:.3f}\t running_accuracy : {:.3f}'
                                            .format(phase, epoch, num_epochs, iter, loss.item(), running_loss / iter,
                                                    current_correct / inputs.size(0),
                                                    running_corrects / running_size
                                                    ), 'green'))
                        writer.add_scalar('Validation Accuracy',
                                          running_corrects / running_size,
                                          epoch )
                        writer.add_scalar('Validation Loss',
                                          running_loss / iter,
                                          epoch)
                        writer.flush()
                    else:
                        logger.info(colored('{} : Epoch ( {}/{} ), Iteration {} : iter_loss : {:.3f} running_loss : '
                                            '{:.3f} iter_acc : {:.3f} running_accuracy : {:.3f} Learning rate : {:.5f}'
                                            .format(phase, epoch, num_epochs, iter, loss.item(),
                                                    running_loss / iter,
                                                    current_correct / inputs.size(0),
                                                    running_corrects / running_size,
                                                    scheduler.get_lr()[0]
                                                    ), 'green'))

                        writer.add_scalar('Training Accuracy',
                                          running_corrects / running_size,
                                          epoch )
                        writer.add_scalar('Training Loss',
                                          running_loss / iter,
                                          epoch)
                        writer.flush()
                iter += 1

            logger.info(colored('{} Epoch ( {}/{} ) : Average Loss : '
                                '{:.3f}\t Average Accuracy : {:.3f}'
                                .format(phase, epoch, num_epochs, running_loss / iter,
                                        running_corrects / running_size
                                        ), 'red'))

            scheduler.step()

            # deep copy the model
            if phase == 'valid':
                if running_corrects / running_size > best_acc:
                    best_acc = running_corrects / running_size
                    logger.info('Saving the model : validation accuracy is {:.3f}'.format(best_acc))
                    best_model_wts = copy.deepcopy(model.state_dict())
                    model.load_state_dict(best_model_wts)
                    torch.save(model, PATH)
                else:
                    logger.info('No need to save for epoch {} as it does not have the best accuracy.'.format(epoch))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), PATH)
    return model

# Train the model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=num_epochs)
writer.close()
logger.info('Process finished.')
print("\nSaving the model...")
torch.save(model_ft, PATH)
'''
Sample run: python train.py --mode=finetue
'''
