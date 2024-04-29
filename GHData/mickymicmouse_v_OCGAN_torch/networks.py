
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import math

##
#network의 가중치를 초기화 해주는 것
#batch norm 의 경우와 일반적인 conv_layer 의 경우
def weights_init(mod):

    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
# target output size of 5x7

# ###
class get_encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, args):
        latent=args.latent
        super(get_encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, stride=2)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(64, 128, 5, stride=2)
        self.batch_norm_1 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 5, stride=2)
        self.batch_norm_2 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, latent, 5, stride=2)

    def forward(self, input): 

        output = self.leaky_relu(self.conv1(input))  
        output = self.batch_norm_1(self.conv2(output))
        output = self.leaky_relu(output)
        output = self.batch_norm_2(self.conv3(output))

        output = self.leaky_relu(output)
        output = torch.tanh(self.conv4(output))
        output = output.view(output.size(0),-1)



        return output

##


class get_decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self,args):
        latent = args.latent
        super(get_decoder, self).__init__()

        self.conv1 = nn.ConvTranspose2d(latent, 256, 5, stride=2)
        self.batch_norm_1 = nn.BatchNorm2d(256)

        self.conv2 = nn.ConvTranspose2d(256, 128, 5, stride=2)
        self.batch_norm_2 = nn.BatchNorm2d(128)  

        self.conv3 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.batch_norm_3 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(p=0.5)
        self.conv4 = nn.ConvTranspose2d(64, 3, 5, stride=2)

 

    def forward(self, input, args):
        latent = args.latent
        output = input.view(input.size(0),latent,1,1)
        output = self.batch_norm_1(self.conv1(output))
        output = F.relu(output)
        output = self.batch_norm_2(self.conv2(output))
        output = F.relu(output)

        output = self.batch_norm_3(self.conv3(output))
        output = self.dropout(output)
        output = F.relu(output)
        output = torch.tanh(self.conv4(output))
        
        return output


class get_disc_latent(nn.Module):
    """
    DISCRIMINATOR latent NETWORK
    """

    def __init__(self,args):
        latent = args.latent
        super(get_disc_latent, self).__init__()
        self.dense_1 = nn.Linear(latent, 128)

        self.dense_2 = nn.Linear(128, 64)

        self.dense_3 = nn.Linear(64, 32)

        self.dense_4 = nn.Linear(32, 16)

        self.dense_5 = nn.Linear(16, 1)
    

    def forward(self, input):

        output = input.view(input.size(0),-1)
        output = F.relu(self.dense_1(output))
        output = F.relu(self.dense_2(output))
        output = F.relu(self.dense_3(output))
        output = F.relu(self.dense_4(output))
        output = self.dense_5(output)
        output = torch.sigmoid(output)
        return output

class get_disc_visual(nn.Module):
    """
    DISCRIMINATOR vision  NETWORK
    """

    def __init__(self):
        super(get_disc_visual, self).__init__()
        self.conv1 = nn.Conv2d(3,12,5,stride=2)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(12,24,5,stride=2)
        self.batch_norm_1 = nn.BatchNorm2d(24)

        self.conv3 = nn.Conv2d(24,48,5,stride=2)
        self.batch_norm_2 = nn.BatchNorm2d(48)

        self.conv4 = nn.Conv2d(48,1,5,stride=2)

    def forward(self, input):

        output = self.conv1(input)
        output = self.leaky_relu(output)

        output = self.batch_norm_1(self.conv2(output))
        output = self.leaky_relu(output)

        output = self.batch_norm_2(self.conv3(output))
        output = self.leaky_relu(output)

        output = self.conv4(output)
        output = torch.sigmoid(output)
        output = output.view(output.size(0),-1)

        return output


class get_classifier(nn.Module):
    """
    Classfier NETWORK
    """

    def __init__(self):
        super(get_classifier, self).__init__()
        self.conv1 = nn.Conv2d(3,64,5,stride=2)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(64,128,5,stride=2)
        self.batch_norm_1 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128,256,5,stride=2)
        self.batch_norm_2 = nn.BatchNorm2d(256)
                
        self.conv4 = nn.Conv2d(256,1,5,stride=2)        


    def forward(self, input):
        output = self.conv1(input)
        output = self.leaky_relu(output)

        output = self.batch_norm_1(self.conv2(output))
        output = self.leaky_relu(output)

        output = self.batch_norm_2(self.conv3(output))
        output = self.leaky_relu(output)

        output = self.conv4(output)

        output = torch.sigmoid(output)
        output = output.view(output.size(0), -1)


        return output