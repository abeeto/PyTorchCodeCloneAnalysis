import numpy as np
from tensor import Tensor
from core import LayerCore


class ReLU(LayerCore):
    def __init__(self):
        super(ReLU, self).__init__()
    def __call__(self, x):
        return np.clip(x, a_min = 0.)

class Sigmoid(LayerCore):
    def __init__(self):
        super(Sigmoid, self).__init__()
    def __call__(self, x):
        return x.sigmoid()

class Tanh(LayerCore):
    def __init__(self):
        super(Tanh, self).__init__()
    def __call__(self, x):
        return x.tanh()

class CrossEntropy(LayerCore):
    def __init__(self):
        super(CrossEntropy, self).__init__()
    def __call__(self, input, target):
        
        pass

class Linear(LayerCore):
    def __init__(self, n_inputs, n_outputs):
        super(Linear, self).__init__()
        W = np.random.rand(n_inputs, n_outputs) * np.sqrt(2.0 / n_inputs)
        self.weight = Tensor(W, requires_grad = True)
        self.bias = Tensor(np.zeros(n_outputs), requires_grad = True)
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)
    def __call__(self, input: Tensor) -> Tensor:
        n_samples = input.data.shape[0]
        return input.mm(self.weight) + self.bias.expand(repeat = n_samples, axis = 0)
        

class MSELoss(LayerCore):
    def __init__(self):
        super(MSELoss, self).__init__()
    def __call__(self, pred, target):
        return ((pred - target) * (pred - target)).sum(0)

class CrossEntropyLoss(LayerCore):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    def __call__(self, input, target):
        return input.cross_entropy(target)