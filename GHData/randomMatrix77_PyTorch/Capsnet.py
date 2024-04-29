import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os

class ConvLayer(nn.Module):

    def __init__(self):

        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(3, 128, 9, 1)

    def forward(self, x):

        out = self.conv(x)
        return out

class PrimaryCaps(nn.Module):

    def __init__(self):

        super(PrimaryCaps, self).__init__()

        self.caps = nn.ModuleList([nn.Conv2d(128, 8, 9, 2) for _ in range(32)])
        self.eps = 1e-7

    def squash(self, x):

        square = (x**2).sum(-1, keepdim = True)
        norm = x / (torch.sqrt(square + self.eps) + self.eps)
        factor = square / (1 + square)
        out = factor * norm
        return out

    def forward(self, x):

        out = [cap(x) for cap in self.caps]
        out = torch.stack(out, dim = 1)
        out = out.view(x.size(0), 32*4*4, -1)
        out = self.squash(out)
        return out

class DigitCaps(nn.Module):

    def __init__(self):

        super(DigitCaps, self).__init__()

        self.W = nn.Parameter(torch.randn(1, 32*4*4, 10, 16, 8))
        self.bij = Variable(torch.zeros(1, 32*4*4, 10, 1))
        self.softmax = nn.Softmax()
        self.eps = 1e-7

    def squash(self, x):

        square = (x**2).sum(-1, keepdim = True)
        norm = x / (torch.sqrt(square + self.eps) + self.eps)
        factor = square / (1 + square)
        out = factor * norm
        return out

    def forward(self, x):

        x = x.view(1, 32*4*4, 8, 1)
        x = torch.stack([x]*10, dim = 2)
        out1 = torch.matmul(self.W, x)
        bij = Variable(torch.zeros(1, 32*4*4, 10, 1))
        print(out1.shape)
        print(bij.shape)

        for i in range(3):

            cij = self.softmax(self.bij)
            cij = cij.unsqueeze(4)

            sij = torch.matmul(cij, out1.transpose(3, 4))
            sij = torch.sum(sij, dim = 1, keepdim = True)
            sij = sij.transpose(3,4)

            vij = self.squash(sij)
            
            if i < 2:

                v_ij = torch.cat([vij]*(32*4*4), dim = 1)

                aij = torch.matmul(out1.transpose(3,4), v_ij)
                aij = aij.unsqueeze(dim = 4)

                bij = bij + aij
                
        return vij

class CapsNet(nn.Module):

    def __init__(self):

        super(CapsNet, self).__init__()

        self.conv = ConvLayer()
        self.pri = PrimaryCaps()
        self.digit = DigitCaps()

    def forward(self, x):

        out = self.conv(x)
        out = self.pri(out)
        out = self.digit(out)

        return out

inp = torch.randn(1, 3, 24, 24)
a = ConvLayer()
b = PrimaryCaps()
c = DigitCaps()

