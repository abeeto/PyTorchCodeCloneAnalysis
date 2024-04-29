# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 17:20:15 2018

@author: vprayagala2

PyTorch Challenge - Classify IRIS Flowers

The project is broken down into multiple steps:

Load and preprocess the image dataset
Train the image classifier on your dataset
Use the trained classifier to predict image content

"""
#%%

import os

import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from torch import utils
import Network as N
#%%
#Check if GPU is available
if torch.cuda.is_available():
    print("CUDA is available")
    device = "cuda"
else:
    print("CUDA is not available, using cpu")
    device = "cpu"
#%%
#1. Load the Data
image_dir = os.path.join("C:\\Data\\Images\\","flower_data")
# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 100
# percentage of training set to use as validation
valid_size = 0.2

# convert data to a normalized torch.FloatTensor
image_size = (255,255)
crop_size=244
random_transforms = [
                    #transforms.ColorJitter(brightness=0.2, contrast=0.6, saturation=0.4, hue=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(torch.randint(low=0,high=360,size=(1,)).item())
                        ]
train_transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(crop_size),
                                transforms.RandomApply(random_transforms, p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485,0.456,0.406),
                                                   (0.229,0.224,0.225))
                                ])

transform = transforms.Compose([transforms.Resize(image_size),
                                #transforms.Grayscale(),
                                transforms.CenterCrop(crop_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485,0.456,0.406),
                                                   (0.229,0.224,0.225))
                                ])

# choose the training and test datasets
train_dir = os.path.join(image_dir,"train")
test_dir = os.path.join(image_dir,"valid")

train_data = datasets.ImageFolder(train_dir,transform=train_transform)
train_loader = utils.data.DataLoader(train_data,batch_size=batch_size, 
                                         shuffle = True,num_workers=num_workers)

validation_data = datasets.ImageFolder(test_dir,transform=transform)
validation_loader = utils.data.DataLoader(validation_data,batch_size=batch_size, 
                                        shuffle = True,num_workers=num_workers)

print("Total number of train Images:%d"%(len(train_loader.dataset)))
print("Total number of test Images:%d"%(len(validation_loader.dataset)))
#%%
#Load the classes
with open(os.path.join(image_dir,"cat_to_name.json"),'r') as f:
    classes=json.load(f)
#%%
# helper function to un-normalize and display an image
#def imshow(img):
#    img = img / 2 + 0.5  # unnormalize
#    plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
#%%
def build_model(pre_train_name=""):
    """
        Create a NN Topology using pre-trained model or new network from scratch
        parameters:
            input:
                pre_train_name : pretrained network name like VGG, RESNET,etc
    """
    pre_train_name = pre_train_name.upper()
    
    if pre_train_name == 'VGG':
        model = models.vgg19(pretrained=True)
    elif pre_train_name == 'RESNET':
        model = models.resnet50(pretrained=True)
    elif pre_train_name == 'INCEPTION':
        model = models.inception_v3(pretrained=True)
    else:
        model = N.Network()
        
    if pre_train_name == 'VGG':
       #turn off gradients
        for param in model.parameters():
            param.requires_grad = False
    
        classifier = nn.Sequential(nn.Linear(25088,4096),
                               nn.ReLU(),
                               nn.Dropout(p=0.4),
                               nn.Linear(4096,2048),
                               nn.ReLU(),
                               nn.Dropout(p=0.4),
                               nn.Linear(2048,512),
                               nn.ReLU(),
                               nn.Dropout(p=0.4),
                               nn.Linear(512,102)
                                )
        model.classifier = classifier
    
    if (pre_train_name == 'RESNET'
    or  pre_train_name == 'INCEPTION'):
       #turn off gradients
        for param in model.parameters():
            param.requires_grad = False
    
        classifier = nn.Sequential(nn.Linear(2048,512),
                               nn.ReLU(),
                               nn.Dropout(p=0.4),
                               nn.Linear(512,102)
                                )
        model.fc = classifier 
    
    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.classifier.parameters(),lr=0.003)
    print(model)
    
    return model, criterion, optimizer

def train(model,criterion,optimizer,train_loader):
    """
        Create a NN Topology using pre-trained model or new network from scratch
        parameters:
            input:
                model : model to train
                criterion : loss function
                optimizer : optimizer to minmize the loss
                train_loader : Train data iterator
            output:
                training loss
    """
    train_loss = 0.0
        
    model.train()
    for data, target in train_loader:
        # move tensors to GPU if CUDA is available
        if device == 'cuda':
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)
        
    return train_loss 


def validate(model,criterion,optimizer,validation_loader):
    """
        Create a NN Topology using pre-trained model or new network from scratch
        parameters:
            input:
                model : model to train
                criterion : loss function
                optimizer : optimizer to minmize the loss
                validation_loader : validation data iterator
            output:
                validation loss
    """
    valid_loss = 0
    model.eval()
    for data, target in validation_loader:
        # move tensors to GPU if CUDA is available
        if device == 'cuda':
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    return valid_loss
    
def save_model(path,model,optimizer,epoch,loss,class_index):
    """
        Save the model to specified path
        parameters:
            input:
                model

    """    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'class_idx':class_index
            }, path)

def load_model(path,model):
    """
        Load the model from given path
        parameters:
            input:
                path
                model
            output:
                model, criterion, optimizermodel, criterion, optimizer, epoch, loss, class_index
    """   
    model, criterion, optimizer = build_model('VGG')
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    class_index = checkpoint['class_idx']
    return model, criterion, optimizer, epoch, loss, class_index

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path)
    img = torch.from_numpy(img)
    img = transform(img)
    #Display Image
    imshow(img)
    
    #prediction
    img.unsqueeze_(0)  #Add Batch dimension
    
    if device == 'cuda':
        img, model  = img.cuda(), model.cuda()
    
    output = model(img)
    # convert output probabilities to predicted class
    pred_prob, pred_label = output.topk(topk)    
    
    return pred_prob, pred_label
      
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open('hopper.jpg')
    im_np = np.asarray(im)
    return im_np
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
#%%
#Sanity check for train images
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx],ax,title=classes[str(labels[idx].numpy().item())])
    #ax.set_title(classes[str(labels[idx].numpy().item())])

#%%
# specify loss function (categorical cross-entropy)
save_path = os.path.join(os.getcwd(),'model_cifar.pt')
model, criterion, optimizer=build_model('VGG')

model.to(device)
print(model)
#%%
#Train the model, validate and caluclate the losses
# number of epochs to train the model
n_epochs = 3

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = train(model,criterion,optimizer,train_loader)
    valid_loss = validate(model,criterion,optimizer,validation_loader)
    
    # calculate average losses
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(validation_loader.dataset)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        class_index=train_loader.dataset.class_to_idx
        save_model(save_path,model,optimizer,epoch,valid_loss,class_index)
        valid_loss_min = valid_loss
#%%
model, criterion, optimizer, epoch, loss, class_index = load_model(save_path)

image_path = os.path.join(os.getcwd(),"test_img")

probs, classes = predict(image_path, model)
print(probs)
print(classes)





