# model.py

import torch.nn as nn
# import pretrainedmodels
import torchvision


def get_model(pretrained=False):
    convnet = torchvision.models.resnet18(pretrained=pretrained)
    convnet.fc = nn.Linear(512, 6)
    return convnet

    # resnet18 = pretrainedmodels.__dict__["resnet18"]
    # if pretrained:
    #     model = resnet18(pretrained="imagenet")
    # else:
    #     model = resnet18(pretrained=None)
    # model.last_linear = nn.Sequential(
    #     nn.BatchNorm1d(512),
    #     nn.Dropout(p=0.25),
    #     nn.Linear(in_features=512, out_features=2048),
    #     nn.ReLU(),
    #     nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
    #     nn.Dropout(p=0.5),
    #     nn.Linear(in_features=2048, out_features=6),
    # )
    # return model