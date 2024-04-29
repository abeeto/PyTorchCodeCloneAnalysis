
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# keep the device info, cuda for gpu use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



@torch.no_grad()
def evaluate(model, val_loader):
    '''
    evaluate a model on val_loader data
    '''
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    '''
    return current learning rate of an optimizer
    '''
    for param_group in optimizer.param_groups:
        return param_group['lr']

def accuracy(outputs, labels):
    '''
    simple function to calculate accuracy
    '''
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class get_bit_representation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sign(nn.ReLU()(x)) # ReLU return 0 if neg and x if not, sign returns 0 if 0 and 1 if pos

    @staticmethod
    def backward(ctx, grad_output): # custom grad identity function of the threshold
        x, = ctx.saved_tensors
        return grad_output


class get_sign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (-1) ** torch.sign(nn.ReLU()(x)) # returns 1 if 0 ( positif ) -1 if 0 ( negatif )

    @staticmethod
    def backward(ctx, grad_output): # custom grad identity function of the threshold
        x, = ctx.saved_tensors
        return -grad_output


def init_weight_bits(shape):
    a = np.sqrt(2 / np.prod(shape[:-1]))  # He standard deviation
    nbits = shape[0]
    probs = a * np.random.normal(0, 1, shape)   # get bit distribution

    # check exactly zero initialization ( proba <= 0 )
    # linear
    if len(shape) == 3:
        for i in range(shape[1]):
            for j in range(shape[2]):
                while np.all(probs[:-1, i, j] <= 0): # if all bits are 0 re calculate
                    probs[:-1, i, j] = a * np.random.normal(0, 1, nbits - 1)

    # conv
    if len(shape) == 5:
        for in_channels in range(shape[3]):
            for out_channels in range(shape[4]):
                for i in range(shape[1]):
                    for j in range(shape[2]):
                        while np.all(probs[:-1, i, j, in_channels, out_channels] <= 0): # if all bits are 0 re calculate
                            probs[:-1, i, j, in_channels, out_channels] = a * np.random.normal(0, 1, nbits - 1)

    return probs


def get_factor(k, target):
    current_std = np.std(k)

    if current_std == 0:
        print("standard deviation can't be zero")
        return 1

    ampl = 1
    eps = 0.001
    min = 0
    max = ampl

    steps = 0
    while np.abs(current_std - target) / target > eps:
        qk = k * ampl
        current_std = np.std(qk)

        if current_std > target:
            max = ampl
            ampl = (max + min) / 2
        elif current_std < target:
            min = ampl
            ampl = (max + min) / 2
        steps += 1

    return ampl


def get_float_from_bits(signfunction, maskfunction, magnitude_block, sign_bit):
    """
    returns the flaot value of the kernel
    """
    if len(magnitude_block) == 0:
        magnitude = 1
    else:
        magnitude = 0
        for i in range(len(magnitude_block)):   # for each magniture block we calculate the base 10 representation and sum them up
            magnitude += maskfunction.apply(magnitude_block[i]) * (2 ** i)
    # make kernel
    kernel = signfunction.apply(sign_bit) * magnitude # dont forget to multiply by the sign
    return kernel


def get_sparsity(k):
    """
    returns the number of negative, zero and positive weights
    """
    neg = np.count_nonzero(k < 0)
    zeros = np.count_nonzero(k == 0)
    pos = np.count_nonzero(k > 0)

    return neg, zeros, pos


def getNZP(net):

    """
    returns the number of negative, zero and positive of a network net
    """
    nsum = 0
    zsum = 0
    psum = 0

    for l in net.modules(): # for each module

        if isinstance(l, Conv2dBit) or isinstance(l, LinearBit): # if its an instance of Conv2Bit or Linear Bit
            neg, zero, pos = l.get_nzp()    # get its nzp
            nsum += neg
            zsum += zero
            psum += pos

    return nsum, zsum, psum