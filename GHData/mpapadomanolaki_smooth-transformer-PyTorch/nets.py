import torch
import torch.nn as nn
from smoothTransformer import smoothTransformer2D

class DisplNet(nn.Module):
    def __init__(self,ch_in=6):
        super(DisplNet,self).__init__()

        self.conv1 = nn.Conv2d(ch_in, 32, kernel_size=3, stride=1, dilation=1, padding=1)
        self.relu1 = nn.LeakyReLU()
        self.conv2 =  nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=1, padding=1)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, dilation=2, padding=2)
        self.relu3 = nn.LeakyReLU()

        self.Up_conv4 = nn.Conv2d(96, 32, kernel_size=3, stride=1, padding=2, dilation=2)
        self.Up_relu4 = nn.LeakyReLU()
        self.Up_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.Up_relu3 = nn.LeakyReLU()
        self.Up_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.Up_relu2 = nn.LeakyReLU()

#        self.affine_conv = nn.Conv2d(16, 9, kernel_size=3, stride=1, padding=1) ##init??
        self.affine_avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.affine_dense = nn.Linear(96,9)

        self.deformable_layer = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)

    def encoder(self, x):
        e1 = self.relu1((self.conv1(x)))
        e2 = self.relu2((self.conv2(e1)))
        e3 = self.relu3((self.conv3(e2)))
        return e1,e2,e3

    def decoder(self,e):
        d4 = self.Up_relu4((self.Up_conv4(e)))
        d3 = self.Up_relu3((self.Up_conv3(d4)))
        d2 = self.Up_relu2((self.Up_conv2(d3)))
        return d2

    def aff(self, out_d):
        #affine_1 = self.affine_conv(out_d)
        affine = self.affine_avgpool(out_d)
        affine = self.affine_dense(affine.squeeze())
        return affine

    def deform(self, out_d):
        deformable_1 = self.deformable_layer(out_d)
#        deformable_1 = self.sigmoid(deformable_1)
        return deformable_1

    def forward(self, moving, reference):

        input = torch.cat( (moving, reference), 1)

        e1,e2,e3 = self.encoder(input)
        e = torch.cat( (e1,e2,e3), 1) #240 planes
        e = self.dropout(e)
        d = self.decoder(e)

        affine = self.aff(e)

        deformable = self.deform(d)
        deformed = smoothTransformer2D([moving, deformable, affine.squeeze()])

        return deformable, affine, deformed
