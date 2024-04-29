import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import os
import torch.optim as optim
import argparse

# models
from models.LeNet import LeNet
from models.AlexNet import AlexNet
from models.VGG import vgg
from models.GoogLeNet import GoogLeNet
from models.ResNet import resnet34, resnet101
from models.MobilNet import MobileNetV2


def build_network(args):
    if args.net.lower() == 'lenet':
        net = LeNet()
    elif args.net.lower() == 'alexnet':
        net = AlexNet(num_classes=args.num_classes, init_weights=True)
    elif args.net.lower() == 'vgg':
        net = vgg('vgg16', num_classes=5, init_weights=True)
    elif args.net.lower() == 'hooglenet':
        net = GoogLeNet(num_classes=args.num_classes, aux_logits=True, init_weights=True)
    elif args.net.lower() == 'resnet43':
        net = resnet34(num_classes=args.num_classes, include_top=True)
    elif args.net.lower() == 'resnet101':
        net = resnet101(num_classes=args.num_classes, include_top=True)
    elif args.net.lower() == 'mobilenet':
        net = MobileNetV2(num_classes=args.num_classes)

    return net
