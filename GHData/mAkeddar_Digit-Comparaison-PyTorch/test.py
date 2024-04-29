#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:11:52 2020
@course: 559DL
@file  : MP N#1
@author: 261344, 261864, 260496
"""

import sys #sys is solely used to deal with parameters passed to script on exec
import torch
from torch import optim
import  torch.nn as nn
from torch.nn import functional as F
import dlc_practical_prologue as prologue

###########################
###########################
#Set script argument here, if cannot pass one through command line
# 0 or no argument = run best network (Weight sharing + aux)
# 1 - Run second best
# 2 - Run "bad" network
# 3 - Run all networks (bad then WS then WS + AuxLoss)
DEFAULT_SCRIPT_ARG = 0
TEST_PERF_ESTIMATE = False #REPEAT LEARNING PROCESS 10 TIMES?
DISCOUNT_FACT_AUXLOSS = 1.0
###########################
###########################

if len(sys.argv) > 1 and sys.argv[1].isnumeric():
    option = int(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2].isnumeric():
        if int(sys.argv[2]) == 1:
            TEST_PERF_ESTIMATE = True
        else:
            TEST_PERF_ESTIMATE = False
    if len(sys.argv) > 3:
        DISCOUNT_FACT_AUXLOSS = float(sys.argv[3])
else:
    option = DEFAULT_SCRIPT_ARG
if 0 < option > 3: option = 0 #any other argument will just run best network

###########################
######### GLOBAL PARAMETERS
N = 1000                #size of train and test sets
eta = 0.1               #learning rate
mini_batch_size = 100   
nEpochs = 25
nbrounds = 10
criterion = nn.CrossEntropyLoss()

###########################
######### INIT
train_input, train_target, train_classes, test_input, test_target, test_classes = \
    prologue.generate_pair_sets(N)
mean, std = train_input.mean(), train_input.std()
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

#Permute randomly train samples
idx=torch.randperm(train_input.shape[0])
train_input=train_input[idx]
train_target=train_target[idx]
train_classes=train_classes[idx]

###########################
######### Basic "bad" network
class ConvNet(nn.Module):
    def __init__(self, arg=None):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2),\
                                    nn.ReLU(),\
                                    nn.MaxPool2d(kernel_size=2, stride=2)\
                                    )#out_put size of 32
    
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),\
                                   nn.ReLU(),\
                                   nn.MaxPool2d(kernel_size=2, stride=2))#input size of 32
        #Digit recognition(output of layer 2 = 64*4*4)
        self.fc1 = nn.Linear(1024,40)
        #comparator
        self.fc2 = nn.Linear(40,2)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0),-1)
        out = F.relu(self.fc1(out.view(-1,1024)))
        out = self.fc2(out)
        return out

###########################
######### Class that make weight sharing net or weight sharing + aux loss net
class ConvNetWS(nn.Module):
    def __init__(self, isAuxLoss = False):
        super(ConvNetWS, self).__init__()

        #dimension and dropout rate for convnet part
        d_OUT_lay1 = 16
        d_OUT_lay2 = 32
        dr_2d_p = 0.3

        #dimension and dropout rate for linear layers part
        d_OUT_fc0 = 100
        d_OUT_fc1 = 10
        dr_1d_p = 0.5
        
        self.isAuxLoss = isAuxLoss
        
        self.layer1 = nn.Sequential(nn.Conv2d(1, d_OUT_lay1, kernel_size=5, stride=1, padding=2),\
                                    nn.ReLU(),\
                                    nn.MaxPool2d(kernel_size=2, stride=2)\
                                    )
        self.bn1 = nn.BatchNorm2d(d_OUT_lay1)
        
        self.layer2 = nn.Sequential(nn.Conv2d(d_OUT_lay1, d_OUT_lay2, kernel_size=4, stride=1, padding=2),\
                                   nn.ReLU(),\
                                   nn.MaxPool2d(kernel_size=2, stride=2)\
                               )
        self.drop2d = nn.Dropout2d(p=dr_2d_p)
        self.bn2 = nn.BatchNorm2d(d_OUT_lay2)
        
        #Linear layers:
        self.fc0 = nn.Linear(d_OUT_lay2*4*4,d_OUT_fc0) # deal with flattened image
        self.dropfc1 = nn.Dropout(p=dr_1d_p)

        self.bnfc1 = nn.BatchNorm1d(d_OUT_fc0)
        self.fc1 = nn.Linear(d_OUT_fc0,d_OUT_fc1) #build digits        
        self.fc2 = nn.Linear(2*d_OUT_fc1,2) #comparator output
        
        
    def forward(self, x):        
        #We can also use torch.split for x, whatever!
        #Treating the images
        A = self.bn2(self.drop2d(self.layer2(self.bn1(self.layer1(x[:,0:1,:,:])))))
        B = self.bn2(self.drop2d(self.layer2(self.bn1(self.layer1(x[:,1:2,:,:])))))
        
        #Digits recognition
        A = F.relu(self.fc1(self.bnfc1(F.relu(self.dropfc1(self.fc0(A.reshape(A.size(0), -1)))))))
        B = F.relu(self.fc1(self.bnfc1(F.relu(self.dropfc1(self.fc0(B.reshape(B.size(0), -1)))))))
            
        #cat is OK with autograd per Pytorch doc
        #Difference of each digit, output accordingly
        out = torch.cat((A,B), 1)
        out = self.fc2(out)
        
        if not self.isAuxLoss:
            return out
        else:
            return out,(A,B)

def compute_nb_errors(model, data_input, data_target, mini_batch_size=mini_batch_size,auxLoss=False):
    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        if auxLoss:
            output, (_,_) = model(data_input.narrow(0, b, mini_batch_size))
        else:
            output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors
        
def train_model(model, train_input, train_target, mini_batch_size):
    optimizer = optim.SGD(model.parameters(), lr = eta)
    for e in range(0, nEpochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if (not TEST_PERF_ESTIMATE):
            print ('e %s - Total loss = %.8f'%(e,sum_loss))

###########################
######### Train model that have auxiliary losses
def train_model_aux(model, train_input, train_target, train_classes, mini_batch_size):
    optimizer = optim.SGD(model.parameters(), lr = eta)
    for e in range(0, nEpochs):
        sum_loss = 0
        aux_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output, (nA, nB) = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            loss_nA = criterion(nA, train_classes[:,0].narrow(0, b, mini_batch_size))
            loss_nB = criterion(nB, train_classes[:,1].narrow(0, b, mini_batch_size))
            sum_loss = sum_loss + loss.item() + loss_nA.item() + loss_nB.item()
            aux_loss = aux_loss +  loss_nA.item() + loss_nB.item()
            model.zero_grad()
            loss.backward(retain_graph=True)
            loss_nA.backward(retain_graph=True)
            loss_nB.backward()
            optimizer.step()
#        for b in range(0, train_input.size(0), mini_batch_size):
#            output, (nA, nB) = model(train_input.narrow(0, b, mini_batch_size))
#            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
#            loss_nA = criterion(nA, train_classes[:,0].narrow(0, b, mini_batch_size))
#            loss_nB = criterion(nB, train_classes[:,1].narrow(0, b, mini_batch_size))
#            sum_loss = sum_loss + loss.item() + DISCOUNT_FACT_AUXLOSS*(loss_nA.item() + loss_nB.item())
#            aux_loss = aux_loss +  loss_nA.item() + loss_nB.item()
#            totloss = loss + DISCOUNT_FACT_AUXLOSS*(loss_nA + loss_nB)
#            model.zero_grad()
#            totloss.backward()
#            optimizer.step()
        if (not TEST_PERF_ESTIMATE):
            print ('e %s - Total loss = %.8f --- AuxLoss = %.8f'%(e,sum_loss,aux_loss))

######################################################################
######################################################################
######### MAIN PROGRAM

#Handles the arg given to the script and execute accordingly, default is else
if option == 1:#Weight sharing solely
    mdlstrlst = ['Weight sharing net']
    modelList = [ConvNetWS]
    argAL = [False]
elif option == 2: #Baseline network
    mdlstrlst = ['Baseline network']
    modelList = [ConvNet]
    argAL = [False]
elif option == 3:
    print("All three models")
    mdlstrlst = ['Baseline network','Weight sharing net','Weight sharing + Aux losses net (alpha=%.2f)'%DISCOUNT_FACT_AUXLOSS]
    modelList = [ConvNet,ConvNetWS,ConvNetWS]
    argAL = [False, False, True]
else:  #Final solution (default), Weight sharing + Aux Loss
    mdlstrlst = ['Weight sharing + Aux losses net (alpha=%.2f)'%DISCOUNT_FACT_AUXLOSS]
    modelList = [ConvNetWS]
    argAL = [True]

for (mdl,auxLoss,mdlstr) in zip(modelList,argAL,mdlstrlst):
    sum_te=[]
    sum_tr=[]
    
    print('\n'+mdlstr+':')
    if TEST_PERF_ESTIMATE:
        print("Round: ",end='')
    for i in range(nbrounds):
        model = mdl(auxLoss)    

        model.train()
        if auxLoss:
            train_model_aux(model,train_input,train_target,train_classes,mini_batch_size)
        else:
            train_model(model, train_input, train_target, mini_batch_size)

        err_train = compute_nb_errors(model,train_input,train_target,mini_batch_size,auxLoss)

        model.train(False)
        err_test = compute_nb_errors(model,test_input,test_target,mini_batch_size,auxLoss)
        sum_te.append(err_test)
        sum_tr.append(err_train)
         
        if TEST_PERF_ESTIMATE:
            print('.',end='')
        else:
            break
    ###endfor rounds
    sum_tr = torch.FloatTensor(sum_tr)
    sum_te = torch.FloatTensor(sum_te)
    print("\nMean train error: %.3f" %sum_tr.mean().item())
    if TEST_PERF_ESTIMATE:
        print("Standard deviation (train): %.3f" %sum_tr.std().item())
    print("Mean test error: %.3f" %sum_te.mean().item())
    if TEST_PERF_ESTIMATE:
        print("Standard deviation (test): %.3f" %sum_te.std().item())
    else:
        nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Number of parameters in the model: %s" %nb_params)
    print('-------')
##endfor mdl