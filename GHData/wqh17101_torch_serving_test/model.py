#!/usr/bin/env python
# _*_coding:utf-8_*_
"""
@Time   :  2022/5/8 19:15
@Author :  Qinghua Wang
@Email  :  597935261@qq.com
"""
import torch
import torch.nn as nn


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def init_model():
    CUDA = torch.cuda.is_available()
    model = NeuralNetwork()
    model.load_state_dict(torch.load("model.pth"))
    if CUDA:
        model = model.cuda()
    model.eval()
    return model
