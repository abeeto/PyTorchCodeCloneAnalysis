# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as vis
from torchsummary import summary

from collections import OrderedDict as OD

import dataset as ds

class CapNet(nn.Module):

    def __init__(self, vocab_size, max_len_cap):
        super(CapNet, self).__init__()

        embed_dim = 256
        hidden_size = 512
        self.embd = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size)
        self.fcn1 = nn.Linear(4096, hidden_size)
        self.fcn2 = nn.Linear(hidden_size, 256)
        self.fcn3 = nn.Linear(256, vocab_size)

    def forward(self, x_img, x_cap):
        x_cap = self.embd(x_cap)
        self.lstm.flatten_parameters()
        x_cap, (a, b) = self.lstm(x_cap)
        x_img = self.fcn1(x_img)
        x_img = F.relu(x_img)
        latent = torch.add(x_cap[-1], x_img)
        x_vec = self.fcn2(latent)
        x_vec = F.relu(x_vec)
        x_vec = self.fcn3(x_vec)
        return x_vec

    def check_updation(self, grad=True):
        for name, p in self.named_parameters():
            print("-------------------------------------------------")
            print("param:")
            print(name, p, sep='\n')
            if grad:
                print("grad:")
                print(name, p.grad, sep='\n')
            print()


def get_features(dir, read=True, download=True):
    if read:
        if download:
            vgg_net = vis.models.vgg16(pretrained="imagenet", progress=True)
        else:
            ## Load model parameters from path
            vgg_net = vis.models.vgg16()
            vgg_net.load_state_dict(torch.load('./models/vgg16-397923af.pth'))

        jpg_files = ds.images_info(dir)

        ## Set requires to eliminate space taken for grads
        for p in vgg_net.parameters():
            p.requires_grad = False

        ## Net architecture
        print(vgg_net)
        # summary(vgg_net, input_size=(3, 224, 224))
        ## Remove the last classifier layer: Softmax
        print("Removing softmax layer of VGG16 ... ")
        vgg_net.classifier = vgg_net.classifier[:-1]
        print(vgg_net)
        # summary(vgg_net, input_size=(3, 224, 224))

        ## Read images with specified transforms
        print("Reading images ... ", end='')
        images = ds.read_image(dir, normalize=True, resize=224, tensor=True)
        print("done.")
        # print(images.keys())
        ## Get feature map for image tensor through VGG-16
        img_featrs = OD()
        print("Gathering images' features from last conv layer ... ", end='')
        for i, jpg_name in enumerate(images.keys()):
            with torch.no_grad():
                print(i, jpg_name)
                img_featrs[jpg_name] = vgg_net(images[jpg_name].unsqueeze(0))
        print("done.")

        return img_featrs