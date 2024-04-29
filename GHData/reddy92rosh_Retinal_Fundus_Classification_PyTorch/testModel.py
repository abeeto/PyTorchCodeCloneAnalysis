#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:32:36 2021

@author: 320131777
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sklearn.metrics

from dataLoader import fundusData, loader 

'''Hyperparameters and User Inputs'''
batchSize = 32 # batch size
inputImageChannels = 3 # number of channels in input images
numClasses = 4 # number of classes/categories
csvLoc = 'Z:/PythonCodes/astrazeneca/csvs/' # locations where csvs are saved
modelLoc = 'Z:/PythonCodes/astrazeneca/trainedModels/' # locations where trained model.pth are saved
modelName = 'train001.pth' # model.pth to be evaluated

def load_checkpoint(filepath):
    '''
    Parameters
    ----------
    filepath (string): 
        Tained model: model.pth

    Returns
    -------
    model : model ready for evaluation
    '''
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model = model.cuda()
    model.eval()
    return model

def testModel(dataloaders, model, device):
    '''
    Test routine
    '''
    model.eval()
    
    CM_N = 0
    CM_C = 0
    CM_DR = 0
    CM_G = 0
    
    with torch.set_grad_enabled(False):
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs.data, 1)
        
            confMat_00 = sklearn.metrics.multilabel_confusion_matrix(labels.cpu(), preds.cpu())

            CM_N += confMat_00[0] # confusion matrix for class: Normal
            CM_C += confMat_00[1] # confusion matrix for class: Cataract
            CM_DR += confMat_00[2] # confusion matrix for class: Proliferative Diabetic Retinopathy
            CM_G += confMat_00[3] # # confusion matrix for class: Glaucoma         
            
        print('-------Normal-------')   
        tn=CM_N[0][0]
        tp=CM_N[1][1]
        fp=CM_N[0][1]
        fn=CM_N[1][0]
        acc=np.sum(np.diag(CM_N)/np.sum(CM_N))
        sensitivity=tp/(tp+fn)
        precision=tp/(tp+fp)
        print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
        print()
        print('Confusion Matirx : ')
        print(CM_N)
        print('- Sensitivity : ',(tp/(tp+fn))*100)
        print('- Specificity : ',(tn/(tn+fp))*100)
        print('- Precision: ',(tp/(tp+fp))*100)
        print('- F1 : ',((2*sensitivity*precision)/(sensitivity+precision))*100)
        print()
        
        print('-------Cataract-------')
        tn=CM_C[0][0]
        tp=CM_C[1][1]
        fp=CM_C[0][1]
        fn=CM_C[1][0]
        acc=np.sum(np.diag(CM_C)/np.sum(CM_C))
        sensitivity=tp/(tp+fn)
        precision=tp/(tp+fp)
        print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
        print()
        print('Confusion Matirx : ')
        print(CM_C)
        print('- Sensitivity : ',(tp/(tp+fn))*100)
        print('- Specificity : ',(tn/(tn+fp))*100)
        print('- Precision: ',(tp/(tp+fp))*100)
        print('- F1 : ',((2*sensitivity*precision)/(sensitivity+precision))*100)
        print()
        
        print('-------Proliferative DR-------')
        tn=CM_DR[0][0]
        tp=CM_DR[1][1]
        fp=CM_DR[0][1]
        fn=CM_DR[1][0]
        acc=np.sum(np.diag(CM_DR)/np.sum(CM_DR))
        sensitivity=tp/(tp+fn)
        precision=tp/(tp+fp)
        print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
        print()
        print('Confusion Matirx : ')
        print(CM_DR)
        print('- Sensitivity : ',(tp/(tp+fn))*100)
        print('- Specificity : ',(tn/(tn+fp))*100)
        print('- Precision: ',(tp/(tp+fp))*100)
        print('- F1 : ',((2*sensitivity*precision)/(sensitivity+precision))*100)
        print()
        
        print('-------Glaucoma-------')
        tn=CM_G[0][0]
        tp=CM_G[1][1]
        fp=CM_G[0][1]
        fn=CM_G[1][0]
        acc=np.sum(np.diag(CM_G)/np.sum(CM_G))
        sensitivity=tp/(tp+fn)
        precision=tp/(tp+fp)
        print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
        print()
        print('Confusion Matirx : ')
        print(CM_G)
        print('- Sensitivity : ',(tp/(tp+fn))*100)
        print('- Specificity : ',(tn/(tn+fp))*100)
        print('- Precision: ',(tp/(tp+fp))*100)
        print('- F1 : ',((2*sensitivity*precision)/(sensitivity+precision))*100)
        print()
        
    return 

dataloaders = {
    'test': loader(fundusData(csvLoc+'test.csv', False), batchSize, shuffle=True)
}
device = torch.device('cuda')  

model = load_checkpoint(modelLoc+modelName)

start = time.time()

test = testModel(dataloaders, model, device)
# print('Val Set IoU: {}'.format(iou))
print('Time: {}'.format(time.time() - start))

