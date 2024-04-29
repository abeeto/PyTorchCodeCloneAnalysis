# -*- coding: utf-8 -*
import torch
from torch import nn
from torchvision import models
from flyai.utils import remote_helper
from efficientnet_pytorch import EfficientNet


def get_net(net_name, load_state=True, train_layer="None"):
    if net_name == 'resnet50':
        path = remote_helper.get_remote_data("https://www.flyai.com/m/resnet50-19c8e357.pth")
        net = models.resnet50(pretrained=False)

        if load_state:
            if path is not None:
                net.load_state_dict(torch.load(path))
            for name, param in net.named_parameters():
                # if 'layer1' in name or 'layer2' in name or 'layer3' in name:
                #     print(name)
                #     param.requires_grad = False
                param.requires_grad = False

        fc_inputs = net.fc.in_features
        net.fc = nn.Linear(fc_inputs, 45, bias=True)
        # print(net.eval())

    elif net_name == 'resnext101_32x8d':
        path = remote_helper.get_remote_data("https://www.flyai.com/m/resnext101_32x8d-8ba56ff5.pth")
        net = models.resnext101_32x8d(pretrained=False)

        if load_state:
            if path is not None:
                net.load_state_dict(torch.load(path))
            for name, param in net.named_parameters():
                # if 'layer1' in name or 'layer2' in name or 'layer3' in name:
                #     print(name)
                #     param.requires_grad = False
                param.requires_grad = False

        fc_inputs = net.fc.in_features
        net.fc = nn.Linear(fc_inputs, 45, bias=True)
        # print(net.eval())

    elif net_name == 'efficientnet-b7':
        path = remote_helper.get_remote_data("https://www.flyai.com/m/efficientnet-b7-dcc49843.pth")
        net = EfficientNet.from_name(net_name)

        if load_state:
            freeze = True
            if path is not None:
                net.load_state_dict(torch.load(path))
            for name, param in net.named_parameters():
                print(name)
                if train_layer in name:
                    freeze = False
                if freeze:
                    param.requires_grad = False
                # param.requires_grad = False

        fc_inputs = net._fc.in_features
        net._fc = nn.Linear(fc_inputs, 45, bias=True)
        # print(net.eval())

    elif net_name == 'efficientnet-b4':
        path = remote_helper.get_remote_data("https://www.flyai.com/m/efficientnet-b4-6ed6700e.pth")
        net = EfficientNet.from_name(net_name)
        # net = EfficientNet.from_pretrained(net_name)

        if load_state:
            freeze = True
            if path is not None:
                net.load_state_dict(torch.load(path))
            for name, param in net.named_parameters():
                print(name)
                if train_layer in name:
                    freeze = False
                if freeze:
                    param.requires_grad = False
                # param.requires_grad = False

        fc_inputs = net._fc.in_features
        net._fc = nn.Linear(fc_inputs, 45, bias=True)
        # print(net.eval())

    else:
        raise ValueError(net_name)
    return net


if __name__ == '__main__':

    # get_net('resnet50')
    # get_net('resnext101_32x8d')
    # get_net('efficientnet-b7')
    get_net('efficientnet-b4')
