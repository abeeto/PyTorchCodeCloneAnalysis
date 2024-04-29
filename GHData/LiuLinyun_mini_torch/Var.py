import numpy as np

class Var():
    def __init__(self, np_ndarray=np.array([0])):
        self.data = np_ndarray
        self.grad = 0
        self.fn = None 
        self.fn_inputs = []
        self.requires_grad = True 

    def backward(self):
        if len(self.data) == 1 and self.fn is not None:
            self.fn.backward(self.fn_inputs, np.ones_like(self.data))
        else:
            raise Exception("不可求非标量的梯度！")
      