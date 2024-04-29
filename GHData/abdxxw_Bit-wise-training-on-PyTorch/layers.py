
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class LinearBit(nn.Module):

    def __init__(self, input_dim, output_dim, config):

        super().__init__()

        self.input_dim = input_dim  # input dimension of the layer
        self.output_dim = output_dim    # output dimension of the layer
        self.default = config["default"]    # layer method, True for default PyTorch, False for bitwise
        self.trainable_bits = config["trainable_bits"]  # list of trainble bits 0 for non trainble and 1 for trainble
        self.nbBits = len(self.trainable_bits)  # number of bits
        self.inference_sequence = config["inference_sequence"]  # bit sign index and magnitude index
        self.name = "linear"    #layer name

        # converts bits prob distribution to binary coefficients
        self.get_bit_representation = get_bit_representation
        self.get_sign = get_sign


        w_shape = (self.input_dim, self.output_dim) #layer weight shape
        w_bit_shape = (self.nbBits,) + w_shape  #bitwise layer weight shape
        print("Building Layer", self.name, w_shape)

        self.std = np.sqrt(2 / np.prod(w_shape[:-1]))   # He standard deviation
        signbit = self.inference_sequence[1]
        magnitudebits = range(self.inference_sequence[0], self.inference_sequence[1])

        if self.default == False:   # in this case we use bitwise weights

            if self.nbBits > 1:
                bit_weights = init_weight_bits(w_bit_shape)   # define the bit weights
            else:
                raise ValueError("number of bits must be greater than 1")


            self.magnitude_block = []
            # for each magnitude bit add it as parameter and specify if its trainable or not
            for i in magnitudebits:
                self.magnitude_block.append(nn.Parameter(torch.tensor(bit_weights[i, ...]).to(device),
                                                         requires_grad=bool(self.trainable_bits[i])))
            self.magnitude_block = nn.ParameterList(self.magnitude_block)
            # also add the signbit
            self.sign_bit = nn.Parameter(torch.tensor(bit_weights[signbit, ...]).to(device),
                                         requires_grad=bool(self.trainable_bits[signbit]))

        else:
            # in case of default layer we use only w ( no bias ) with he kaiming init
            self.weight = nn.Parameter(torch.randn(w_shape))
            nn.init.kaiming_normal_(self.weight, mode='fan_in')

    def forward(self, x):
        # simple linear forward after calling get_weight
        return F.linear(x, self.get_weight().T)

    def get_weight(self):
        if self.default:
            return self.weight
        else:
            # at each call we calculate the weight value from the bit weights
            self.weight = get_float_from_bits(self.get_sign, self.get_bit_representation, self.magnitude_block, self.sign_bit)
            self.alpha = get_factor(self.weight.clone().detach().cpu().numpy(),self.std)  # for good convergence according to paper
            self.weight *= self.alpha
            return self.weight.float().to(device)

    def get_bits(self):
        #return bit tensor
        return self.magnitude_block + [self.sign_bit]

    def get_nzp(self):
        # return negatif / zero / positif weight numbers
        return get_sparsity(self.get_weight())


class Conv2dBit(nn.Module):

    def __init__(self, input_dim, filters, kernel_size, stride, padding, config):

        super().__init__()

        self.input_dim = input_dim # input dimension of the layer
        self.filters = filters  # number of filters of the layer
        self.kernel_size = kernel_size  #kernel size of the layer
        self.stride = stride
        self.padding = padding
        self.default = config["default"]    # layer method, True for default PyTorch, False for bitwise
        self.trainable_bits = config["trainable_bits"]  # list of trainble bits 0 for non trainble and 1 for trainble
        self.nbBits = len(self.trainable_bits)  # number of bits
        self.inference_sequence = config["inference_sequence"]  # bit sign index and magnitude index
        self.name = "conv" #layer name

        # converts bits prob distribution to binary coefficients
        self.get_bit_representation = get_bit_representation
        self.get_sign = get_sign

        k_shape = list((self.kernel_size, self.kernel_size)) + [self.input_dim, self.filters]   # kernel shape

        k_bit_shape = [self.nbBits]
        k_bit_shape.extend(k_shape)

        print("Building Layer", self.name, k_shape)


        self.std = np.sqrt(2 / np.prod(k_shape[:-1])) # He standard deviation
        signbit = self.inference_sequence[1]
        magnitudebits = range(self.inference_sequence[0], self.inference_sequence[1])

        if self.default == False:   # in this case we use bitwise weights

            if self.nbBits > 1:
                bit_kernel = init_weight_bits(k_bit_shape)   # define the bit kernel
            else:
                raise ValueError("number of bits must be greater than 1")


            self.magnitude_block = []
            # for each magnitude bit add it as parameter and specify if its trainable or not
            for i in magnitudebits:
                self.magnitude_block.append(nn.Parameter(torch.tensor(bit_kernel[i, ...]).to(device),
                                                         requires_grad=bool(self.trainable_bits[i])))
            self.magnitude_block = nn.ParameterList(self.magnitude_block)
            # also add the signbit
            self.sign_bit = nn.Parameter(torch.tensor(bit_kernel[signbit, ...]).to(device),
                                         requires_grad=bool(self.trainable_bits[signbit]))

        else:
            # in case of default layer we use only kernel ( no bias ) with he kaiming init
            self.kernel = nn.Parameter(torch.randn(k_shape))
            nn.init.kaiming_normal_(self.kernel, mode='fan_in')

    def forward(self, x):
        return F.conv2d(x, self.get_kernel().T, stride=self.stride, padding=self.padding)

    def get_kernel(self):
        if self.default:
            return self.kernel
        else:
            # at each call we calculate the kernel value from the bit weights
            self.kernel = get_float_from_bits(self.get_sign, self.get_bit_representation, self.magnitude_block, self.sign_bit)
            self.alpha = get_factor(self.kernel.clone().detach().cpu().numpy(),self.std)  # for good convergence according to paper
            self.kernel *= self.alpha
            return self.kernel.float().to(device)

    def get_bits(self):
        #return bit tensor
        return self.magnitude_block + [self.sign_bit]

    def get_nzp(self):
        # return negatif / zero / positif weight numbers
        return get_sparsity(self.get_kernel())
