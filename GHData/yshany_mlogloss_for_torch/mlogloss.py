import torch
import torch.nn as nn
class MLogLoss(nn.Module):
    def __init__(self):
        super(MLogLoss,self).__init__()
    def forward(self,input,target):
        return -torch.sum(torch.log(torch.clamp(input,1e-15,1))*target)
