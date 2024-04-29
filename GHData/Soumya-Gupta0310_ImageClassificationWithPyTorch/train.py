import torch
from torch import nn
from torchvision import datasets, transforms
import torchvision.models as models
import torch.optim as optim
from workspace_utils import active_session
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os.path as path
import json
from collections import Counter
import argparse

def TransformAndLoadImages(data_dir):    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
                   'test': transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
                  }       
    # TODO: Load the datasets with ImageFolder
    image_datasets = {'train_dataset':datasets.ImageFolder(train_dir,transform=data_transforms['train']),
                  'val_dataset':datasets.ImageFolder(valid_dir,transform=data_transforms['test']),
                  'test_dataset':datasets.ImageFolder(test_dir,transform=data_transforms['test'])}

    #TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'trainloader':torch.utils.data.DataLoader(image_datasets['train_dataset'],batch_size=32, shuffle=True),
               'val_loader':torch.utils.data.DataLoader(image_datasets['val_dataset'],batch_size=32,shuffle=True),
               'testloader':torch.utils.data.DataLoader(image_datasets['test_dataset'],batch_size=32,shuffle=True)}
    return data_transforms,image_datasets,dataloaders

# TODO: Initialize model according to architecture
def InitializeModel(architecture): 
    switcher = {         
        'vgg13': models.vgg13(pretrained=True), 
        'vgg16': models.vgg16(pretrained=True),
        'vgg19': models.vgg19(pretrained=True), 
        'resnet18': models.resnet18(pretrained=True),
        'resnet34': models.resnet34(pretrained=True), 
        'resnet50': models.resnet50(pretrained=True), 
        'resnet101': models.resnet101(pretrained=True),
        'resnet152': models.resnet152(pretrained=True), 
        'squeezenet1.0': models.squeezenet1_0(pretrained=True),
        'squeezenet1.1': models.squeezenet1_1(pretrained=True), 
        'densenet121': models.densenet121(pretrained=True),
        'densenet169': models.densenet169(pretrained=True),
        'densenet161': models.densenet161(pretrained=True), 
        'densenet201': models.densenet201(pretrained=True)
    } 
    model=switcher.get(architecture,"Invalid argument for architecture")    
    return model

#Freeze the weights of the pre-trained network
def FreezeWeightsOfPretrainedNetwork(model):        
    for param in model.parameters():
        param.requires_grad=False
    return model

#Construct a feed-forward network to be appeneded to the pre-trained network
def AppendForwardClassifier(numberOfInputNodes,outputNodes,hidden):

    listOfHiddenLayerNodes=ConvertStringInputToList(hidden) #Converting the number of nodes in hidden layer 
                                                            #from string to list of integers    
    feedForwardClassifier=nn.Sequential(nn.Linear(numberOfInputNodes,listOfHiddenLayerNodes[0]),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(listOfHiddenLayerNodes[0],listOfHiddenLayerNodes[1]),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(listOfHiddenLayerNodes[1],listOfHiddenLayerNodes[2]),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(listOfHiddenLayerNodes[2],outputNodes),                        
                            nn.LogSoftmax(dim=1))    
    return feedForwardClassifier


def ConvertStringInputToList(stringToConvert):
    listOfHiddenLayerNodes=stringToConvert.replace('[','').replace(']','').split(',')
    listOfHiddenLayerNodes=list(map(lambda x:int(x),listOfHiddenLayerNodes))    
    return listOfHiddenLayerNodes


#Check if cuda is available or not and set torch.device accordingly
def GetDevice(gpu):
    device=None
    if(gpu==False):
        device=torch.device('cpu') 
    else:
        isCudaAvailable=torch.cuda.is_available()    
        if(isCudaAvailable==True):
            device=torch.device('cuda')
        else:
            device=torch.device('cpu') 
    return device

def GetNumberOfOutputClasses(image_datasets):
    return len(image_datasets['train_dataset'].class_to_idx)

#Build Complete Model
def BuildCompleteModel(outputNodes,architecture,hidden,gpu):
    model=InitializeModel(architecture)
    model=FreezeWeightsOfPretrainedNetwork(model)  
    if(architecture.startswith('vgg')):
        numberOfNodesInInputLayer=model.classifier[0].weight.shape[1]
        model.classifier=AppendForwardClassifier(numberOfNodesInInputLayer,outputNodes,hidden)
        feedForwardClassifier=model.classifier
    elif(architecture.startswith('densenet')):
        numberOfNodesInInputLayer=model.classifier.weight.shape[1]
        model.classifier=AppendForwardClassifier(numberOfNodesInInputLayer,outputNodes,hidden)
        feedForwardClassifier=model.classifier
    elif(architecture.startswith('squeezenet')):
        numberOfNodesInInputLayer=model.classifier[1].weight.shape[1]
        model.classifier=AppendForwardClassifier(numberOfNodesInInputLayer,outputNodes,hidden)
        feedForwardClassifier=model.classifier
    else: #for resnet
        numberOfNodesInInputLayer=model.fc.weight.shape[1]
        model.fc=AppendForwardClassifier(numberOfNodesInInputLayer,outputNodes,hidden)
        feedForwardClassifier=model.fc
    device=GetDevice(gpu)
    model=model.to(device)
    return model,feedForwardClassifier


#Define the loss function to be used
def DefineLossFunctionAndOptimizer(feedForwardClassifier, learning_rate):    
    criterion = nn.NLLLoss()
    #Build the optimizer that will update the weights of the classifier
    optimizer=optim.Adam(feedForwardClassifier.parameters(), lr=learning_rate)    
    return criterion,optimizer

train_losses=[]
val_losses=[]
accuracies=[]

# Train the feed-forward network
def TrainNetwork(model,criterion,optimizer,dataloaders,epochs,gpu):    
    device=GetDevice(gpu) 
    with active_session():
        for epoch in range(epochs):
            train_loss=0
            val_loss=0
            accuracy=0
            for images,labels in dataloaders['trainloader']:
                images,labels=images.to(device),labels.to(device)        
                optimizer.zero_grad()
                logps=model.forward(images)
                loss=criterion(logps,labels)
                loss.backward()
                train_loss+=loss.item()
                optimizer.step()
            else:
                with torch.no_grad():
                    for images,labels in dataloaders['val_loader']:
                        images,labels=images.to(device), labels.to(device)
                        #Turn off dropout while applying model on test dataset
                        model.eval()
                        logps=model.forward(images)
                        loss=criterion(logps,labels)
                        val_loss+=loss.item()
                        #Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(train_loss/len(dataloaders['trainloader']))
            val_losses.append(val_loss/len(dataloaders['val_loader']))
            accuracies.append(accuracy/len(dataloaders['val_loader']))
            model.train()
        print('train_losses : '+str(train_losses))
        print('val_losses : '+str(val_losses))
        print('accuracy : '+str(accuracies))   


def CreateFilesIfNotExist(modelCheckPoint, optimizerCheckPoint):
    if(path.exists(modelCheckPoint)==False):
        f=open(modelCheckPoint,'x')
        f.close()
    if(path.exists(optimizerCheckPoint)==False):
        f=open(optimizerCheckPoint,'x')
        f.close()
        
def SaveCheckPoints(architecture,hidden,model,optimizer,modelCheckPoint,optimizerCheckPoint):
    model_info={'architecture':architecture,'hidden_layers':hidden,'state_dict':model.state_dict(),'class_to_idx':image_datasets['train_dataset'].class_to_idx}
    optimizer_info={'state_dict':optimizer.state_dict(),'n_epochs':len(train_losses)}
    torch.save(model_info,modelCheckPoint)
    torch.save(optimizer_info,optimizerCheckPoint)

if(__name__=='__main__'):
    parser=argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--save_dir")
    parser.add_argument("--arch")
    parser.add_argument("--learning_rate")
    parser.add_argument("--hidden_units")
    parser.add_argument("--epochs")
    parser.add_argument("--gpu",action='store_true') #the value of this option will be true when the --gpu argument is used but if no value is passed for it
    args=parser.parse_args()
    
    data_dir=args.data_dir
    save_dir=args.save_dir
    gpu=False #Setting the default value of gpu
    if(args.gpu!=None):
        gpu=args.gpu
    architecture=None
    if(args.arch!=None):
        architecture=args.arch
    else:
        architecture='vgg13' #Setting the default value of architecture
    learning_rate=None
    if(args.learning_rate!=None):
        learning_rate=float(args.learning_rate)
    else:
        learning_rate=0.001 #Setting the default value of learning_rate
    hidden=None
    if(args.hidden_units!=None):
        hidden=args.hidden_units
    else:
        hidden='[1536,1024,512]' #Setting the default value of hidden_units
    epochs=None
    if(args.epochs!=None):
        epochs=int(args.epochs)
    else:
        epochs=5 #Setting the default value of epochs
    
    data_transforms,image_datasets,dataloaders=TransformAndLoadImages(data_dir)
    outputNodes=GetNumberOfOutputClasses(image_datasets)
    model,feedForwardClassifier=BuildCompleteModel(outputNodes,architecture,hidden,gpu)    
    criterion,optimizer=DefineLossFunctionAndOptimizer(feedForwardClassifier,learning_rate)
   
    train_losses=[]
    val_losses=[]
    accuracies=[]
    
    TrainNetwork(model,criterion,optimizer,dataloaders,epochs,gpu)
    if(save_dir!=None):
        CreateFilesIfNotExist(save_dir+'/modelCheckPoint.pth',save_dir+'/optimizerCheckPoint.pth')
        SaveCheckPoints(architecture,hidden,model,optimizer,save_dir+'/modelCheckPoint.pth',save_dir+'/optimizerCheckPoint.pth')



