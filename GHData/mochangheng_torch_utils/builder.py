import torch
import torch.nn as nn

def get_activation(name):
    activations = {
        'none': nn.Identity(),
        'relu': nn.ReLU(),
        'silu': nn.SiLU(),
        'leaky_relu': nn.LeakyReLU(),
    }
    return activations[name]

def build_fc(in_features, out_features, batch_norm=False, activation='none'):
    fc = nn.Linear(in_features, out_features)
    activ = get_activation(activation)

    if batch_norm:
        norm = nn.BatchNorm1d(in_features)
        return nn.Sequential(norm, fc, activ)

    return nn.Sequential(fc, activ)

def build_conv(in_channels, out_channels, kernel_size, stride=1, activation='none', batch_norm=False):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride)
    activ = get_activation(activation)

    if batch_norm:
        norm = nn.BatchNorm2d(in_channels)
        return nn.Sequential(norm, activ, conv)

    return nn.Sequential(conv, activ)

def build_pool(pool_size, kernel_size=3, mode='max'):
    if mode == 'max':
        pool = nn.MaxPool2d(kernel_size, stride=pool_size, padding=kernel_size//2)
    else:
        raise NotImplementedError("Not implemented.")
    return pool
