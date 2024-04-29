import torch
import torch.nn as nn
import numpy as np


def canon_conv(in_maps, out_maps, kernel_size, stride=1, padding=1, dilation=1):
    return nn.Conv2d(
        in_channels=in_maps,
        out_channels=out_maps,
        kernel_size=kernel_size,
        stride=stride,
        padding=1,
        dilation=dilation)


def max_pool(kernel_size, stride=None, padding=0, dilation=1):
    return nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        
        dilation=1)


def frac_conv(in_maps, out_maps, kernel_size, stride=2, padding=0, dilation=1):
    return nn.ConvTranspose2d(
        in_channels=in_maps,
        out_channels=out_maps,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
        dilation=1)


def conv_one(in_maps, out_maps, stride=1, padding=0, dilation=1):
    return nn.Conv2d(
        in_channels=in_maps,
        out_channels=out_maps,
        kernel_size=1,
        stride=stride,
        padding=padding,
        dilation=dilation)



class conv_conv(nn.Module):
    def __init__(self, in_maps, out_maps, kernel_size):
        super(conv_conv, self).__init__()
        
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.kernel_size = kernel_size

        self.conv_a = canon_conv(self.in_maps, self.out_maps, self.kernel_size)
        self.conv_b = canon_conv(self.out_maps, self.out_maps, self.kernel_size)
        
    def forward(self, tensor):
        tensor = nn.functional.relu(self.conv_a(tensor))
        tensor = nn.functional.relu(self.conv_b(tensor))
        return tensor

    
class pool_conv_conv(nn.Module):
    def __init__(self, in_maps, out_maps, kernel_size, pool_size):
        super(pool_conv_conv, self).__init__()
        
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.kernel_size = kernel_size
        self.pool_size = pool_size

        self.conv_a = canon_conv(self.in_maps, self.out_maps, self.kernel_size)
        self.conv_b = canon_conv(self.out_maps, self.out_maps, self.kernel_size)
        self.pool = max_pool(self.pool_size, self.pool_size) 
        
    def forward(self, tensor):
        tensor = self.pool(tensor)
        tensor = nn.functional.relu(self.conv_a(tensor))
        tensor = nn.functional.relu(self.conv_b(tensor))
        return tensor

    
class conv_conv_one(nn.Module):
    def __init__(self, in_maps, out_maps, kernel_size, class_num):
        super(conv_conv_one, self).__init__()
        
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.kernel_size = kernel_size
        self.class_num = class_num

        self.conv_a = canon_conv(self.in_maps, self.out_maps, self.kernel_size)
        self.conv_b = canon_conv(self.out_maps, self.out_maps, self.kernel_size)
        self.conv_c = conv_one(self.out_maps, self.class_num)
        
    def forward(self, tensor):
        tensor = nn.functional.relu(self.conv_a(tensor))
        tensor = nn.functional.relu(self.conv_b(tensor))
        tensor = nn.functional.sigmoid(self.conv_c(tensor))
        return tensor

    
class up_copy(nn.Module):
    def __init__(self, in_maps, out_maps, up_kernel_size):
        super(up_copy, self).__init__()
        
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.class_num = class_num
        self.up_kernel_size = up_kernel_size
        
        self.conv_a = frac_conv(self.in_maps, self.out_maps, self.up_kernel_size)
        
    def forward(self, tensor_a, tensor_b):
        tensor = self.conv_a(tensor_a)
        tensor = torch.cat((tensor, tensor_b), 1)
        return tensor    


class up_copy_conv(nn.Module):
    def __init__(self, in_maps, out_maps, kernel_size, up_kernel_size):
        super(up_copy_conv, self).__init__()
        
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size

        self.conv_a = frac_conv(self.in_maps, self.out_maps, self.up_kernel_size)
        self.conv_b = canon_conv(self.in_maps, self.out_maps, self.kernel_size)
        self.conv_c = canon_conv(self.out_maps, self.out_maps, self.kernel_size)
        
    def forward(self, tensor_a, tensor_b):
        tensor = self.conv_a(tensor_a)
        tensor = torch.cat((tensor, tensor_b), 1)
        tensor = nn.functional.relu(self.conv_b(tensor))

        tensor = nn.functional.sigmoid(self.conv_c(tensor))
        return tensor


class NeuralNetwork(nn.Module):
    def __init__(self, layer_list, kernel_size, up_kernel_size, pool_size):
        super(NeuralNetwork, self).__init__()
        
        self.layer_list = layer_list
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.pool_size = pool_size

        self.down_funs = []
        self.up_funs = []

        maps = self.layer_list[0]
        self.layer_list = self.layer_list[1:]
        in_maps = self.input_maps(self.layer_list)
        
        self.in_fun = conv_conv(maps, in_maps, self.kernel_size)
        
        for in_maps, out_maps in zip(self.layer_list, self.layer_list[1:]):
            fun = pool_conv_conv(in_maps, out_maps, self.kernel_size, self.pool_size)
            self.down_funs.append(fun)

        rev_a = reversed(self.layer_list[1:])
        rev_b = reversed(self.layer_list[:-1])
            
        for in_maps, out_maps in zip(rev_a, rev_b):
            fun = up_copy_conv(in_maps, out_maps, self.kernel_size, self.up_kernel_size)
            self.up_funs.append(fun)

        self.out_fun = conv_one(self.layer_list[0], 1)
        
            
    @staticmethod
    def list_checker(self, layer_list):
        if self.layer_list < 4:
            raise ValueError('Not enough layers given')

    def input_maps(self, layer_list):
        return self.layer_list[0]


    def forward(self, input_tensor):
        tensor = self.in_fun(input_tensor)
        down_tensors = []
        down_tensors.append(tensor)
        
        for fun in self.down_funs:
            tensor = fun(tensor)
            down_tensors.append(tensor)

        rev_c = reversed(down_tensors[:-1])
        
        for fun, old_tens in zip(self.up_funs, rev_c):
            tensor = fun(tensor, old_tens)

        tensor = self.out_fun(tensor)
        
        print(np.shape(nn.ParameterList))
        return tensor
