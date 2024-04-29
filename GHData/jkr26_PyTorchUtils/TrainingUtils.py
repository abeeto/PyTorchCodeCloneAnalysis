#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#import time
import torch
import torch.nn as nn
from datetime import datetime
#from numpy import random
#import torch.nn as nn
import pdb
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorboardX

"""
Created on Mon May  7 19:02:45 2018

@author: jkr
"""

use_cuda = torch.cuda.is_available()
torch.backends.cudnn.enabled = True
now = datetime.now()

class QRStep(nn.Module):
    """(One step of) the classical QR algorithm for eigenvector/eigenvalue
    computation implemented in PyTorch.
    """
    def __init__(self):
        super(QRStep, self).__init__()
        
    def _GSStep(col, previous_cols):
        """col assumed to be a 1d tensor, previous_cols
        a normalized list of 1d tensors
        """
        to_subtract = torch.zeros(col.shape)
        for c in previous_cols:
            to_subtract = to_subtract + torch.dot(col, c) * c
        col = col - to_subtract
        return col/(torch.dot(col, col)**(1/2))
        
    def _normalQRStep(curr):
        #Compute Q via Gram-Schmidt
        curr[:, 0] = curr[:, 0]/(torch.dot(curr[:, 0], curr[:, 0])**(1/2))
        previous_cols = [curr[:, 0]]
        for k in range(1, curr.shape[1]):
            curr[:, k] = QRStep._GSStep(curr[:, k], previous_cols)
            previous_cols.append(curr[:, k])
        #Return Q
        return curr
    
    def forward(self, init_map, op):
        """init_map should be the product of all the Qs computed thus far.
        
        op should be the operator we are looking to GD over
        """
        #This is written under the assumption 
        curr = torch.mm(torch.mm(init_map.permute(1, 0).contiguous(), op), init_map)
        currQ = QRStep._normalQRStep(curr)
        return_map = torch.mm(init_map, currQ)
        return return_map

class implicitQRStep(nn.Module):
    """Classical QR algorithm for eigenvector/eigenvalue computation implemented
    in PyTorch.
    """
    def __init__(self):
        super(implicitQRStep, self).__init__()
        
    def _implicitQRStep(curr):
        #Compute QR decomposition
        #Return Q 
        pass
    
    def forward(init_map, op):
        #This is written under the assumption 
        curr = torch.mm(init_map, op)
        currQ = implicitQRStep._implicitQRStep(curr)
        return_map = torch.mm(currQ.permute(1, 0).contiguous(), init_map)
        return return_map
        

def to_np(x):
    return x.data.cpu().numpy()

def saveModel(problem_name, model, iterate):
    pass

def setOptimizerLr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def sgdr(period, batch_idx):
    radians = math.pi*float(batch_idx)/period
    return 0.5*(1.0 + math.cos(radians))

def plot_lr_finder(model, criterion, data, batch_size):
    lr_list = []
    loss_list = []
    lr = 1e-15
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    m = np.Inf
    train, train_response = data[0], data[1]
    n = len(train)
    t = 0
    n_iters = int(1e4)
    while lr < 100:
        total_loss = 0
        for b in range(0, n_iters, batch_size):
            batch =  (torch.Tensor(train.loc[b % n:(b+batch_size) %  n, :].values), torch.Tensor(train_response.loc[b % n:(b+batch_size) % n].values))
            if batch[0].size()[0] > 0:
                if use_cuda:
                    batch = (batch[0].cuda(), batch[1].cuda())
                preds = model(batch[0].unsqueeze(1))
                loss = criterion(preds.squeeze(1), batch[1])
                total_loss += loss.item()
                loss.backward()
                opt.step()
        t = total_loss/(n_iters/batch_size)
        print(str(t))
        lr_list.append(np.log10(lr))
        loss_list.append(t)
        lr = 2*lr
        setOptimizerLr(opt, lr)
        if t < m:
            m = t
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(np.array(lr_list), np.array(loss_list))
    plt.show()

def batchedTrainIters(name, n_iters, batch_size, print_every, model, optimizer, criterion,  data, text, use_sgdr=False, period=100):
    """Reusable trainiters function for convNet.
    Args:
    n_iters, int, num iterations
    batch_size, int, batch size
    model, subclass of torch.nn.Module
    optimizer, subclass of torch.nn.Optim
    criterion, loss function
    data, 3-tuple of 2-tuples pd.DataFrames, consisting of:
        (train, train_response), (val, val_response),  (test, test_response)
    """
    if use_cuda:
        model = model.cuda()
    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    writer = tensorboardX.SummaryWriter('/media/jkr/hdd1/TensorboardLogs/'+name+'/'+now)
    train, train_response = data[0][0], data[0][1]
    val, val_response = data[1][0], data[1][1]
    test, test_response = data[2][0], data[2][1]
    min_val_loss = np.Inf
    assc_test_loss = np.Inf
    n = len(train)
    total = 0
    orig_lr = optimizer.param_groups[0]['lr']
    for b in range(0, n_iters, batch_size):
        if use_sgdr:
            if b % period == 0:
                setOptimizerLr(optimizer,orig_lr*sgdr(period, b))
        if (b+batch_size) %  n > b % n:
            optimizer.zero_grad()
            variables = (torch.Tensor(train.loc[b % n:(b+batch_size) %  n, :].values), torch.Tensor(train_response.loc[b % n:(b+batch_size) % n].values))
            if use_cuda:
                variables = (variables[0].cuda(), variables[1].cuda())
            output = model(variables[0].unsqueeze(1))
            loss = criterion(output.squeeze(1), variables[1])
            total += loss.item()
            loss.backward()
            optimizer.step()
            if b % print_every == 0 and b > 0:
                print('Loss total for the past '+str(print_every)+' examples is '+str(total))
                writer.add_scalar('train_loss', total, b)
                writer.add_text('hyperparams', text)
                for tag, value in model.named_parameters():
                    try:
                        tag = tag.replace('.', '/')
                        writer.add_histogram(tag, to_np(value), b)
                        writer.add_histogram(tag+'/grad', to_np(value.grad), b)
                    except AttributeError:
                        pdb.set_trace()
                total = 0
                total_val_loss = 0
                for k in range(0, len(val), batch_size):
                    val_vars = (torch.Tensor(val.loc[k:k+batch_size, :].values), torch.Tensor(val_response.loc[k:k+batch_size].values))
                    if use_cuda:
                        val_vars = (val_vars[0].cuda(), val_vars[1].cuda())
                    val_output = model(val_vars[0].unsqueeze(1))
                    val_loss = criterion(val_output.squeeze(1), val_vars[1])
                    total_val_loss += val_loss.item()
                print('Val loss is '+str(total_val_loss))
                writer.add_scalar('val_loss', total_val_loss, b)
                # Embedding a little hard to pull off
#                writer.add_embedding(val_output, metadata=val_vars[1])
                if total_val_loss < min_val_loss:
                    min_val_loss = total_val_loss
                    total_test_loss = 0
                    for j in range(0, len(test), batch_size):
                        test_vars = (torch.Tensor(test.loc[j:j+batch_size, :].values), torch.Tensor(test_response.loc[j:j+batch_size].values))
                        if use_cuda:
                            test_vars = (test_vars[0].cuda(), test_vars[1].cuda())
                        test_output = model(test_vars[0].unsqueeze(1))
                        test_loss = criterion(test_output.squeeze(1), test_vars[1])
                        total_test_loss += test_loss.item()
                    print('Test loss is '+str(total_test_loss))
                    assc_test_loss = total_test_loss
                    writer.add_scalar('test_loss', assc_test_loss, b)
    return total_test_loss