import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, in_channel = 3, out_channel = 64):
        super(Discriminator, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size = 5, stride = 2, padding = 2),
                                   nn.LeakyReLU(0.2))
        
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel, 2 * out_channel, kernel_size = 5, stride = 2, padding = 2),
                                   nn.LeakyReLU(0.2))
        
        self.conv3 = nn.Sequential(nn.Conv2d(2 * out_channel, 4 * out_channel, kernel_size = 5, stride = 2, padding = 2),
                                   nn.LeakyReLU(0.2))
        
        self.conv4 = nn.Sequential(nn.Conv2d(4 * out_channel, 8 * out_channel, kernel_size = 5, stride = 2, padding = 2),
                                   nn.LeakyReLU(0.2))
        
        self.conv5 = nn.Sequential(nn.Conv2d(8 * out_channel, 8 * out_channel, kernel_size = 5, stride = 2, padding = 2),
                                   nn.LeakyReLU(0.2))

        self.batchnorm1 = nn.BatchNorm2d(out_channel)
        self.batchnorm2 = nn.BatchNorm2d(2 * out_channel)
        self.batchnorm3 = nn.BatchNorm2d(4 * out_channel)
        self.batchnorm4 = nn.BatchNorm2d(8 * out_channel)
        self.batchnorm5 = nn.BatchNorm2d(8 * out_channel)

        self.fc = nn.Linear(8 * 8 * 8 * out_channel, 1)

    def forward(self, condition, real_fake):

        x = torch.cat([condition, real_fake], dim = 0)

        x = self.conv1(x)
        x = self.batchnorm1(x)
        #print (x.size())
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        #print (x.size())
                
        x = self.conv3(x)
        x = self.batchnorm3(x)
        #print (x.size())

        x = self.conv4(x)
        x = self.batchnorm4(x)
        #print (x.size())

        x = self.conv5(x)
        x = self.batchnorm5(x)
        #print (x.size())

        x = torch.reshape(x, [-1, 8 * 8 * 8 * self.out_channel])
        #print (x.size())

        x = self.fc(x)
        #print (x.size())

        out = torch.reshape(x, [-1])

        return out



def weights_init(m):
    if isinstance(m ,nn.Conv2d):
        init.uniform_(m.weight, -0.02*np.sqrt(3), 0.02*np.sqrt(3))
        init.constant_(m.bias, 0)