import numpy as np

'''
    Defines Cores for Layers
'''
class LayerCore(object):
    def __init__(self):
        self.parameters = list()

    def get_parameters(self):
        return self.parameters


'''
    Defines the Arithmetic Operations of Tensor
'''
class TensorCore(object):
    def __init__(self, data, dtype = float):
        
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        self.data = self.data.astype(dtype)


    # Arithmetic Operations
    def __add__(self, other):
        return self.data + other.data
    def __neg__(self):
        return -1 * self.data
    def __sub__(self, other):
        return self.data - other.data
    def __mul__(self, other):
        return self.data * other.data
    def mm(self, other):
        return self.data @ other.data
    def conv(self, other):
        pass

    # Utility Functions
    def shape(self):
        return self.data.shape    
    def astype(self, dtype):
        return self.data.astype(dtype)
    def swapaxis(self, axis1, axis2):
        return np.swapaxes(self.data, axis1 = axis1, axis2 = axis2)
    def moveaxis(self, source, destination):
        return np.moveaxis(self.data, source = source, destination = destination)
    def transpose(self, axes: list = None):
        return np.transpose(a = self.data, axes = axes)
    def sum(self, axis):
        return np.sum(self.data, axis = axis)
    def expand_dims(self, axis):
        return np.expand_dims(self.data, axis = axis)
    def repeat(self, repeat, axis):
        return np.repeat(self.data, repeats = repeat, axis = axis)
    def expand(self, repeat, axis):
        return np.repeat(
            np.expand_dims(
                self.data, 
                axis = axis
            ),
            repeats = repeat,
            axis = axis
        )

    # Activation Functions
    def sigmoid(self):  return 1. / (1 + np.exp(-self.data))
    def tanh(self):     return np.tanh(self.data)

'''
    Function Collections for Computing Gradients
'''

class grad_engine(TensorCore):
    def __init__(self):
        pass

    def _gradient_add(self, grad_val):
        pass


