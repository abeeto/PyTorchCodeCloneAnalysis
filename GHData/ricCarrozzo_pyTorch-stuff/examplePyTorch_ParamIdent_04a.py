# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 10:53:34 2022

@author: RC
"""
# example of pytorch-based parameter identification

# load the environment

import sys
import matplotlib.pyplot as plt
import numpy as np
import math

import torch

class warmUpTemp(torch.nn.Module):
    def __init__(hehe, A0):
        """
        2 parameters
        
        needed for T = B *( 1 - exp( -x/C ) )
        
        """
        super().__init__()
        # hehe.A = torch.nn.Parameter(A0[0])
        hehe.B = torch.nn.Parameter(A0[0])
        hehe.C = torch.nn.Parameter(A0[1])
        
    def forward(hehe, x):
        """
        Calculate the model
    
        """
        # return hehe.A + hehe.B * ( 1.0 - ( torch.exp(-x/hehe.C) ) )
        return hehe.B * ( 1.0 - ( torch.exp(-x/hehe.C) ) )
    
    def string(hehe):
        """
        
        Print out Model

        """
        # return f'y = {hehe.A.item()} + {hehe.B.item()} * exp( -x/{hehe.C.item()} )'
        return f'y = {hehe.B.item()} * exp( -x/{hehe.C.item()} )'
    
# prepare the data

inpFrq = 0.1
smplTime = 0.1
sigLen = 1300
piConst = math.pi
xx = np.arange( 0, sigLen )
timeAxis = smplTime * torch.tensor( xx )
sinA = torch.sin( 2*piConst*inpFrq*timeAxis )
# tempA = 53.5 + 45.8 * ( 1.0 - torch.exp( -timeAxis/13.4 )  )
tempA = 45.8 * ( 1.0 - torch.exp( -timeAxis/13.4 )  )
sigA = tempA+5*sinA

# iniPar = [ 53.0, 30.0, 20 ]
iniPar = [ 80.0, 10.0 ]
myModel = warmUpTemp( torch.tensor( iniPar ) )

#criter = torch.nn.MSELoss(reduction='sum')
criter = torch.nn.MSELoss()

# optim = torch.optim.SGD(myModel.parameters(), lr=1e-6)
# optim = torch.optim.Rprop(myModel.parameters(), lr=1e-6)
optim = torch.optim.RMSprop(myModel.parameters(), lr=1e-3)

#
lossFcnPrev = 1e6
#
for k in range(200000):
    out_pred = myModel( timeAxis )
   
    lossFcn = criter( out_pred, sigA )
    #    
    if k % 500 == 499:
        print(k, lossFcn.item(), abs( lossFcnPrev - lossFcn.item()))
    #
    brkThrsh = abs( lossFcnPrev - lossFcn.item() )
    if ( ( brkThrsh < 1e-6 ) and ( brkThrsh > 0.0 )):
        break
    if k > 0:
        lossFcnPrev = lossFcn.item()
    #       
    
    optim.zero_grad()
    lossFcn.backward()
    optim.step()

    
