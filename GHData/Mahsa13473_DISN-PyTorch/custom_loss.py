# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from config import Struct, load_config, compose_config_str

config_dict = load_config(file_path='./config_sdfnet.yaml')
configs = Struct(**config_dict)
batch_size = configs.batch_size

threshold = 0.01


class sdf_loss(torch.nn.Module):

    def __init__(self):
        super(sdf_loss, self).__init__()

    def forward(self, x, y):
        batch_size = x.size(0)
        weight = torch.ones(y.size(0), 1)

        if configs.use_cuda:
            weight = weight.float()#.cuda()

        index = torch.nonzero(y < threshold)[:, 0]

        weight[index] = 4
        weight = torch.t(weight)

        '''
        if y[0] < threshold:
            m = 4
        else:
            m = 1
        '''
        loss = torch.mm(weight, torch.abs(x - y))/batch_size
        return loss


if __name__ == '__main__':
    batch_size = 16
    torch.manual_seed(13)

    loss = sdf_loss()
    # loss = nn.L1Loss()
    input = torch.randn(batch_size, 1, requires_grad=True)
    target = torch.randn(batch_size, 1)
    print(input.shape)
    print(target.shape)
    print(target)
    output = loss(input, target)
    print(output)
