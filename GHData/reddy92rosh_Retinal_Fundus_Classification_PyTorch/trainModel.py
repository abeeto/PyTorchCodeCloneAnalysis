# -*- coding: utf-8 -*-
"""
Training routine
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import pandas as pd
import time

from net import MyVGG11
from dataLoader import fundusData, loader

'''Hyperparameters and User Inputs'''
epochs = 100 # number of epochs
batchSize = 32 # batch size
inputImageChannels = 3 # number of channels in input images
numClasses = 4 # number of classes/categories
learningRate = 0.001 # learning rate
momentum = 0.9 # momentum for SGD optimizer
csvLoc = 'Z:/PythonCodes/astrazeneca/csvs/' # locations where csvs are saved
saveLoc = 'Z:/PythonCodes/astrazeneca/trainedModels/' # location to save trained models
modelName = 'train001.pth' # trained model name to be saved as...

def write_df(metrics_dict, init_df, csv_fname):
    """Insert/append a metrics dict to df and write to csv file
    :param dict metrics_dict: dict with segmentation or detection
     metrics
    :param bool init_df: indicator to write a new df
    :param str csv_fname: fname with full path to save metrics_df
    :return pd.Dataframe metrics_df: with row added
    """

    if init_df:
        metrics_df = pd.DataFrame(columns=metrics_dict.keys(),
                                  dtype=object)
    else:
        metrics_df = pd.read_csv(csv_fname, index_col=0)
    metrics_df = metrics_df.append(metrics_dict, ignore_index=True)
    metrics_df.to_csv(csv_fname)
    return metrics_df


def trainModel(dataloaders, optimizer, model, lossFunc, device):
    '''
    Training routine
    '''
    model.train()
    
    lossEpoch = []
    
    with torch.set_grad_enabled(True):
        for inputs, labels in dataloaders['train']:

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            
            loss.backward()
            optimizer.step()
            # print(pred)
            
            lossEpoch.append(loss.item())
            # print(lossEpoch)

            
        lossEpoch = np.array(lossEpoch)
        
    return np.mean(lossEpoch, axis=0)

def valModel(dataloaders, optimizer, model, lossFunc, device):
    '''
    Validation routine
    '''
    model.eval()
    
    lossEpoch = []
    
    with torch.set_grad_enabled(False):
        for inputs, labels in dataloaders['val']:
           
            inputs = inputs.to(device)
            labels = labels.to(device)
         
            optimizer.zero_grad()
            
            pred = model(inputs)
            loss = criterion(pred, labels)
            # print(loss)
            
            lossEpoch.append(loss.item())
            
        lossEpoch = np.array(lossEpoch)
        
    return np.mean(lossEpoch, axis=0)


dataloaders = {
    'train': loader(fundusData(csvLoc+'train.csv', augmentation = True), batchSize, shuffle=True), 
    'val': loader(fundusData(csvLoc+'val.csv', False), batchSize, shuffle=False)
}

device = torch.device('cuda')  
model = MyVGG11(in_ch = inputImageChannels, num_classes = numClasses)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr = learningRate, momentum = momentum)

criterion = nn.CrossEntropyLoss() # cross entrpy loss

bestValLoss = 1e10
best_model_wts = copy.deepcopy(model.state_dict())

for i in range(epochs):
    start = time.time()
    print('Epoch: {}'.format(i+1))
    trainEpochLoss = trainModel(dataloaders, optimizer, model, criterion, device)
    valEpochLoss = valModel(dataloaders, optimizer, model, criterion, device)
    print('Train Loss: {}'.format(trainEpochLoss))
    print('Val Loss: {}'.format(valEpochLoss))
    print('Time (per epoch): {}'.format(time.time() - start))
    
    epoch_dict = {'train_loss': trainEpochLoss,
                  'val_loss': valEpochLoss}
    init_df = False
    if i == 0:
        init_df = True
    write_df(epoch_dict, init_df, saveLoc+modelName.replace('.pth', '.csv'))
        
        
    '''Saves model with best validation loss'''    
    if valEpochLoss.sum() < bestValLoss:
        print("Saving best model")
        bestValLoss = valEpochLoss.sum()
        best_model_wts = copy.deepcopy(model.state_dict())
        
        print('Best val loss: {}'.format(bestValLoss))
        model.load_state_dict(best_model_wts)
        bestModel = {'model': MyVGG11(in_ch = inputImageChannels, num_classes = numClasses),
              'state_dict': model.state_dict()}
        torch.save(bestModel, saveLoc+modelName)
     






