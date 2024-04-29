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

def InitializeModel(architecture='vgg13'): 
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
def AppendForwardClassifier(numberOfInputNodes,outputNodes,hidden='[1536,1024,512]'):

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
def BuildCompleteModel(architecture,hidden,outputNodes,gpu):    
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


def LoadModel(modelCheckPoint,gpu):    
    model_info=torch.load(modelCheckPoint)
    architecture=model_info['architecture']
    hidden=model_info['hidden_layers']    
    model_state_dict=model_info['state_dict']
    class_to_idx=model_info['class_to_idx']
    outputNodes=len(class_to_idx)
    model,feedforwardClassifier=BuildCompleteModel(architecture,hidden,outputNodes,gpu)
    model.load_state_dict(model_state_dict)
    model.class_to_idx=class_to_idx    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    transform=transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    image=transform(image)    
    return image        


def predict(image_path, model, topk=5):
    
    # TODO: Implement the code to predict the class from an image file
    # TODO: Implement the code to predict the class from an image file
    with torch.no_grad():
        image=Image.open(image_path)
        image=process_image(image)
        device=GetDevice(gpu)
        model=model.to(device)
        image=image.reshape(1,3,224,224)
        image=image.to(device)
        model.eval()    
        logps=model(image)
        ps=torch.exp(logps)
        top_ps,top_classes_index=ps.topk(topk,dim=1)    
        return top_ps, top_classes_index
    
def GetCategoryToNameMapping(file='ImageClassifier/cat_to_name.json'):    
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

# The below function maps the different class indices to their corresponding class labels/integers
def GetClasses(top_classes_index,model):
    idx_to_class={v:k for k,v in model.class_to_idx.items()} #Inverting the class_to_idx dictionary
                                                              # to get idx to class mapping
    top_classes_index=pd.Series(top_classes_index)
    top_classes=top_classes_index.map(idx_to_class).tolist()
    return top_classes

# The below function maps the different class labels/integers to their corresponding class names
def GetClassNames(top_classes,classToNameMapping):    
    top_classes=pd.Series(top_classes)
    top_class_names=top_classes.map(classToNameMapping)
    return top_class_names.tolist()


if(__name__=='__main__'):
    parser=argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("checkpoint")
    parser.add_argument("--top_k")
    parser.add_argument("--category_names")
    parser.add_argument("--gpu",action='store_true')
    args=parser.parse_args()
    
    image_path=args.input
    checkpoint=args.checkpoint
    gpu=False #Setting the default value of gpu
    if(args.gpu!=None):
        gpu=args.gpu
    topk=None
    if(args.top_k!=None):
        topk=int(args.top_k)
    category_names_file=None
    if(args.category_names!=None):
        category_names_file=args.category_names
    
    model=LoadModel(checkpoint,gpu)
    if(topk!=None):
        top_ps,top_classes_index=predict(image_path,model,topk)
    else:
        top_ps,top_classes_index=predict(image_path,model)
    
    top_ps=top_ps.reshape(top_ps.shape[1]).tolist() # Converting the 2d tensor in a list
    top_classes_index=top_classes_index.reshape(top_classes_index.shape[1]).tolist() # Converting the 2d tensor in a list    
    top_classes=GetClasses(top_classes_index,model)
    if(category_names_file!=None):
        cat_to_name=GetCategoryToNameMapping(category_names_file)
    else:
        cat_to_name=GetCategoryToNameMapping()
    top_class_names=GetClassNames(top_classes,cat_to_name) #Getting class names from class integers
    print('top_classes_index:',top_classes_index, 'top_class_ps: ',top_ps)
    print('top_classes_labels:',top_classes,'top_class_names:',top_class_names)
    
    

