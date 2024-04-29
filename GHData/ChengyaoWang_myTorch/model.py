import numpy as np


class Module(object):
    def __init__(self):
        pass


    def __call__(self):
        pass


class Sequential(Module):

    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.sequential_layer = []
        for layer in layers:
            self.sequential_layer.append(layer)
        
    def add(self, layer):
        self.sequential_layer.append(layer)
    
    def get_parameters(self):
        params = []
        for layer in self.sequential_layer:
            params += layer.get_parameters()
        return params
        
    def __call__(self, x):
        for layer in self.sequential_layer:
            x = layer(x)
        return x