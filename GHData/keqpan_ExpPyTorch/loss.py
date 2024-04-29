import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math

class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta, phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss

def PSNR(imgs, gt_imgs):
    batch_size = imgs.size(0)
    PIXEL_MAX = gt_imgs.reshape(batch_size, -1).max(-1)[0] #use groundtruth maximal pixel
    return 10*torch.log10(PIXEL_MAX**2/((imgs.reshape(batch_size, -1) - gt_imgs.reshape(batch_size, -1))**2).mean(-1))
    
# def PSNR(imgs1, imgs2, PIXEL_MAX):
#     batch_size = imgs1.size(0)
#     return 10*torch.log10(PIXEL_MAX**2/((imgs1.reshape(batch_size, -1) - imgs2.reshape(batch_size, -1))**2).mean(-1))