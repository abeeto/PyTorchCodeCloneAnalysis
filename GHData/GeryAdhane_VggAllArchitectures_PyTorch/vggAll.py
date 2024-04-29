#import the necessary packages

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.functional as F 
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import time
import os
import copy
import matplotlib.pyplot as plt

#Load Data
# Data augmentation and normalization for training
# Just normalization for validation
#Define the model
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}
#Assumption is you have train, val, and test folders, where within each folder
#You have Mask and noMask folders as well.

data_dir = 'Mask_NoMask/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                for x in ['train', 'val','test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                            shuffle=True, num_workers=4)
            for x in ['train', 'val','test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
class_names = image_datasets['train'].classes

device = "cuda:0" if torch.cuda.is_available() else "cpu"

#Dictonary of different VGG architectures
#This is the general architectures of all VGG models.
VGGTypes = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    ,'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    ,'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    ,'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    
    }
#Flattening the network and the fully connected models is common for all modesl.
#Flatten the network
#4096X4096X1000 fully connected layer
#For our case we will implement for 2 classes

class VggGeneral(nn.Module):
    def __init__(self, in_channels = 3, num_classes=2, modelc='VGG16'):
        super(VggGeneral, self).__init__()
        self.in_channels = in_channels
        self.modelc = modelc
        if self.modelc == 'VGG11':
            self.conv_layers = self.create_layer(VGGTypes['VGG11'])
        elif self.modelc == 'VGG13':
            self.conv_layers = self.create_layer(VGGTypes['VGG13'])
        elif self.modelc == 'VGG16':
            self.conv_layers = self.create_layer(VGGTypes['VGG16'])
        elif self.modelc == 'VGG19':
            self.conv_layers = self.create_layer(VGGTypes['VGG19'])

        #Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 4096)
            ,nn.ReLU()
            ,nn.Dropout(p = 0.5)

            ,nn.Linear(4096, 4096)
            ,nn.ReLU()
            ,nn.Dropout(p = 0.5)

            ,nn.Linear(4096, num_classes)
        )
    def forward(self, input):
        input = self.conv_layers(input)
        input = input.reshape(input.shape[0], -1)
        input = self.fc(input)

        return input

    def create_layer(self, architecture):

        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                    kernel_size = (3,3), stride=(1,1), padding = (1,1))
                    ,nn.BatchNorm2d(x)
                    ,nn.ReLU(x) 
                ]
                in_channels = x

            elif x == 'M':
                layers += [
                    nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
                ]
        return nn.Sequential(*layers)

#Method to train our modelc
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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

#Method to visualize the model predictions
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

#Method to display image from datasets
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

if  __name__ == "__main__":

    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VggGeneral(in_channels = 3, num_classes = 2, modelc = 'VGG16').to(device)
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # Train the model
    model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
    # Save the model
    torch.save(model,'model.h5')
    #Visualize some results
    visualize_model(model)