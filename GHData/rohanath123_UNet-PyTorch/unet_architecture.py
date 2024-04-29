import torch
import torch.nn as nn
import numpy as np

class Down(nn.Module):
  def __init__(self, input_size, output_size):
    super(Down, self).__init__()

    self.input_size = input_size
    self.output_size = output_size
    
    self.conv1 = nn.Conv2d(in_channels = self.input_size, out_channels = self.output_size, kernel_size = 3, stride = 1, padding = 1)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels = self.output_size, out_channels = self.output_size, kernel_size = 3, stride = 1, padding = 1)
    self.relu2 = nn.ReLU()

    self.maxp = nn.MaxPool2d(2) 

  def forward(self, input):
    mid = self.relu2(self.conv2(self.relu1(self.conv1(input))))
    output = self.maxp(mid)
    return output, mid

class Up(nn.Module):
  def __init__(self, input_size, output_size):
    super(Up, self).__init__()

    self.input_size = input_size
    self.output_size = output_size

    self.up = nn.ConvTranspose2d(in_channels = self.input_size, out_channels = self.output_size, kernel_size = 2, stride = 2)
    self.conv1 = nn.Conv2d(in_channels= self.input_size, out_channels= self.output_size, kernel_size = 3, stride = 1, padding = 1)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels= self.output_size, out_channels= self.output_size, kernel_size = 3, stride = 1, padding = 1)
    self.relu2 = nn.ReLU()
    

  def forward(self, input, mid):
    output = self.up(input)
    print("ConvTranspose:", output.size())
    output = torch.cat([output, mid], dim = 1)
    output = self.relu2(self.conv2(self.relu1(self.conv1(output))))
    print("Convs:", output.size())
    return output

class UNetModel(nn.Module):
  def __init__(self):
    super(UNetModel, self).__init__()

    self.down1 = Down(3, 32)
    self.down2 = Down(32, 64)
    self.down3 = Down(64, 128)
    self.down4 = Down(128, 256)

    self.conv_in1 = nn.Conv2d(256, 512, 3, 1, 1)
    self.relu_in1 = nn.ReLU()
    self.conv_in2 = nn.Conv2d(512, 512, 3, 1, 1)
    self.relu_in2 = nn.ReLU()

    self.up1 = Up(512, 256)
    self.up2 = Up(256, 128)
    self.up3 = Up(128, 64)
    self.up4 = Up(64, 32)

    self.conv_last = nn.Conv2d(32, 1, 3, 1, 1)
    self.sig = nn.Sigmoid()
  
  def forward(self, input):

    out1, mid1 = self.down1(input)
    out2, mid2 = self.down2(out1)
    out3, mid3 = self.down3(out2)
    out4, mid4 = self.down4(out3)

    output = self.relu_in2(self.conv_in2(self.relu_in1(self.conv_in1(out4))))

    output = self.up1(output, mid4)
    output = self.up2(output, mid3)
    output = self.up3(output, mid2)
    output = self.up4(output, mid1)
    
    output = self.sig(self.conv_last(output))
    return output

