import numpy as np


class SGD(object):
    def __init__(self, parameters, lr = 0.2):
        self.parameters = parameters
        self.lr = lr
    
    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0.
    
    def step(self, zero_grad = True):
        for p in self.parameters:
            p.data -= p.grad.data * self.lr
            if zero_grad:
                p.grad.data *= 0.