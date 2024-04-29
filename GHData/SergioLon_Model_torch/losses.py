import torch
#import numpy as np
def nmse(output, target):
    if len(output.size())>1:
        loss=torch.zeros((output.size(1),1))
        for i in range (output.size(1)):
            
            loss[i] = torch.sum((output[:,i] - target[:,i])**2) / torch.sum(target[:,i]**2)
            #print(loss)
        return torch.sum(loss)
    else:
        loss = torch.sum((output - target)**2) / torch.sum(target**2)
        return loss

def nl1(output, target):
    loss = torch.sum(torch.abs(output - target)) / torch.sum(torch.abs(target))
    return loss