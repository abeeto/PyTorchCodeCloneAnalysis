import numpy as np 
from Var import Var

eps = 1e-6

class Mul():
    def __init__(self):
        pass

    def forward(self, x, y):
        z = Var() 
        z.data = np.matmul(x.data, y.data) 
        if x.requires_grad and y.requires_grad:
            z.fn = self 
            z.fn_inputs = [x, y]
        return z
    
    def __call__(self, x, y):
        return self.forward(x, y)
        
        
    def backward(self, inputs, pre_grad):
        x = inputs[0]
        y = inputs[1]
        if x.requires_grad:
            x.grad += np.matmul(pre_grad, y.data.T)
            if x.fn is not None:
                x.fn.backward(x.fn_inputs, x.grad)
        if y.requires_grad:
            y.grad += np.matmul(pre_grad.T, x.data).T
            if y.fn is not None:
                y.fn.backward(y.fn_inputs, y.grad)

mul = Mul()

class Sigmoid():
    def __init__(self):
        pass

    def forward(self, x):
        y = Var() 
        y.data = 1/(1+np.exp(-x.data)) 
        if y.requires_grad:
            y.fn = self 
            y.fn_inputs = [x]
        return y

    def __call__(self, x):
        return self.forward(x)

    def backward(self, inputs, pre_grad):
        x = inputs[0]
        if x.requires_grad:
            s = 1/(1+np.exp(-x.data))
            s = np.multiply(s, 1-s)
            x.grad += np.multiply(pre_grad, s)
            if x.fn is not None:
                x.fn.backward(x.fn_inputs, x.grad)

sigmoid = Sigmoid()

class Add():
    def __init__(self):
        pass 

    def forward(self, x, y):
        z = Var() 
        z.data = np.add(x.data, y.data) 
        if x.requires_grad and y.requires_grad:
            z.fn = self 
            z.fn_inputs = [x, y]
        return z
    
    def __call__(self, x, y):
        return self.forward(x, y)
        
        
    def backward(self, inputs, pre_grad):
        x = inputs[0]
        y = inputs[1]
        if x.requires_grad:
            x.grad += pre_grad
            if x.fn is not None:
                x.fn.backward(x.fn_inputs, x.grad)
        if y.requires_grad:
            y.grad += pre_grad
            if y.fn is not None:
                y.fn.backward(y.fn_inputs, y.grad)

add = Add()

class Reshape():
    def __init__(self):
        pass 

    def forward(self, x, shape):
        y = Var()
        y.data = x.data.reshape(shape)
        if x.requires_grad:
            y.fn = self 
            y.fn_inputs = [x]
        return y

    def __call__(self, x, shape):
        return self.forward(x, shape)

    def backward(self, inputs, pre_grad):
        x = inputs[0]
        if x.requires_grad:
            x.grad += pre_grad.reshape(x.data.shape)
            if x.fn is not None:
                x.fn.backward(x.fn_inputs, x.grad)

reshape = Reshape()
        
class LogLoss():
    def __init__(self):
        pass 

    def forward(self, x, label):
        y = Var()
        y.data = -np.log(abs(x.data)+eps) if label.data == 1 else -np.log(abs(1-x.data)+eps)
        if x.requires_grad:
            y.fn = self 
            y.fn_inputs = [x, label]
        return y

    def __call__(self, x, shape):
        return self.forward(x, shape)

    def backward(self, inputs, pre_grad):
        x = inputs[0]
        label = inputs[1]
        if x.requires_grad:
            x.grad += -pre_grad/(abs(x.data)+eps) if int(label.data) == 1 else -pre_grad/(abs(1-x.data)+eps)
            if x.fn is not None:
                x.fn.backward(x.fn_inputs, x.grad)
        if label.requires_grad:
            label.grad += -np.multiply(pre_grad, np.log(abs(x.data)+eps)) if label.data == 1 else np.multiply(pre_grad, np.log(abs(1-x.data)+eps))
            if label.fn is not None:
                label.fn.backward(label.fn_inputs, label.grad)

logloss = LogLoss()

class MSELoss():
    def __init__(self):
        pass 

    def forward(self, x, y):
        z = Var()
        z.data = (x.data - y.data)**2
        if x.requires_grad and y.requires_grad:
            z.fn = self 
            z.fn_inputs = [x, y]
        return z

    def __call__(self, x, y):
        return self.forward(x, y)

    def backward(self, inputs, pre_grad):
        x = inputs[0]
        y = inputs[1]
        if x.requires_grad:
            x.grad += 2*np.multiply(pre_grad, (x.data - y.data))
            if x.fn is not None:
                x.fn.backward(x.fn_inputs, x.grad)
        if y.requires_grad:
            y.grad += -2*np.multiply(pre_grad, (x.data - y.data))
            if y.fn is not None:
                y.fn.backward(y.fn_inputs, y.grad)

mse_loss = MSELoss()
