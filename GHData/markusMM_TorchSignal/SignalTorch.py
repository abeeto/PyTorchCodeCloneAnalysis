# -*- coding: utf-8 -*-
"""
Signal Processing tools in torch

@author: Markus Meister
"""

import torch
import math

torch.pi = torch.tensor(math.pi)

def MoveLastToFirst(x):
    for xi in range(len(x.shape)-1):
        x = x.transpose(xi,-1)
    return x

def MoveFirstToLast(x):
    for xi in range(len(x.shape)-1):
        x = x.transpose(xi,xi+1)
    return x

def roll(x, n):  
    return torch.cat((x[-n:], x[:-n]))

def shft(x, n):
    n = int(n*(1 - n // x.shape[0]))
    if n == 0:
        return x
    return torch.cat((
            0*x[-n:],
            1*x[:-n],
        ))

def roll_mat(x,order,strafe=1):
    Xe = torch.zeros(order,*x.shape).to(x.device)
    for m in range(order):
        Xe[m] = roll(x,m*strafe)
    return Xe

def shft_mat(x,order,strafe=1):
    Xe = torch.zeros(order,*x.shape).to(x.device)
    for m in range(order):
        Xe[m] = shft(x,m*strafe)
    return Xe

def ConvT(x, filter, strafe = 1):
    '''Inplace convolution function
Does convolve two a filter tensor with a signal tensor along the signal axis.
The output length though will be the the signal length.
The Equation is unnormed:
    $y[n] = sum_k{h[k]x[n-k]}$ with $k\in[0,..,K-1}$ and $n\in[0,..,N-1]$
Here, $K$ is the signal length.
    '''
    t1_flg = 0
    
    Xe = roll_mat(x, filter.shape[-1], strafe)
    Xe = MoveFirstToLast(Xe)
    
    if Xe.shape[0] != filter.shape[0]:
        Xe = torch.ones([filter.shape[0],*Xe.shape]).to(x.device) * Xe[None]
        t1_flg = 1
    
    filter = torch.ones(Xe.shape).to(x.device)*filter[:,None,None]
    
    Xe = (filter * Xe).sum(dim=-1)
    
    if t1_flg:
        Xe = Xe.transpose(0,1)
    
    return Xe

def WinT(ten, win_type='ham', win_coef=None):
    
    if type(win_type) == type(None):
        win_type = 'ham'
    
    if len(win_type) < 3:
        win_type += '   '
    
    if win_type[:3] == 'han':
        if type(win_coef) == type(None):
            win_coef = .50
        win = win_coef * (1 - torch.cos( 2*torch.pi*torch.arange(ten.shape[0]) / ten.shape[0] ))
    if win_type[:3] == 'ham':
        if type(win_coef) == type(None):
            win_coef = .54
        win = win_coef * (1 - torch.cos( 2*torch.pi*torch.arange(ten.shape[0]) / ten.shape[0] ))
    if win_type[:3] == 'bla':
        if type(win_coef) == type(None):
            win_coef = [.16, '.5*(1-a)', '.5', '.5*a']
        a = win_coef[0]
        win = eval(win_coef[1]) 
        rew = torch.arange(ten.shape[0]) / ten.shape[0]
        for c, coef in enumerate(win_coef[1:]):
            win += eval(coef) * torch.cos( 2*(c+1)*torch.pi*rew )
    if win_type[:3] == 'rec':
        if type(win_coef) == type(None):
            win_coef = 1
        win = win_coef * torch.ones(ten.shape[0])
    if win_type[:3] == 'tri':
        if type(win_coef) == type(None):
            win_coef = 1
        win = win_coef * (1 - torch.arange(ten.shape[0]) / ten.shape[0])
    
    
    for i in range(len(ten.shape)-1):
        win = win.unsqueeze(i+1)
    
    return ten * win
    
