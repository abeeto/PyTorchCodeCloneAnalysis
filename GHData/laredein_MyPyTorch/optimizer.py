import layerslib
import numpy as np
from abc import ABCMeta, abstractmethod
class optimiser:
    def __init__(self,lr,momentum,layers):
        self.momentum=momentum
        self.lr=lr
        self.layers=layers
        if len(self.layers)==1:
            self.layers[0].prev=None
            self.layers[0].next=None
        else:
            for i in range(len(self.layers)):
                if i==0:
                    self.layers[i].prev=None
                    self.layers[i].next=self.layers[i+1]
                elif i==(len(self.layers)-1):
                    self.layers[i].prev=self.layers[i-1]
                    self.layers[i].next=None
                else:
                    self.layers[i].next=self.layers[i+1]
                    self.layers[i].prev=self.layers[i-1]
    def add_param_group(self,inp,out,type):
        if type=="linear":
            self.layers.append(layerslib.Linear(inp,out))
        if type=="sigmoid":
            self.layers.append(layerslib.Sigmoid(inp,out))
        if type=="prelu":
            self.layers.append(layerslib.PReLU(inp,out))
        if type=="relu":
            self.layers.append(layerslib.ReLU(inp,out))
        if type=="tanh":
            self.layers.append(layerslib.Tanh(inp,out))
        if type=="logsigmoid":
            self.layers.append(layerslib.LogSigmoid(inp,out))
        if type=="softmax":
            self.layers.append(layerslib.SoftMax(inp,out))
    def load_state_dict(self,momentum,lr):
        self.momentum=momentum
        self.lr=lr
    def state_dict(self):
        return [self.momentum,self.lr,len(self.layers)]
    def zero_grad(self):
        for layer in self.layers:
            if layer.type=="flat" or layer.type=="conv2d" or layer.type=="maxpool2d":
                continue
            else:
                layer.oldgrad = layer.grad
                layer.grad=np.zeros((layer.outsize,layer.inpsize))

    @abstractmethod
    def step(self):
        pass
class SGD(optimiser):
    def step(self):
        for layer in self.layers:
            if layer.type=="flat" or layer.type=="conv2d" or layer.type=="maxpool2d":
                continue
            elif layer.type == "conv2d":
                layer.arrk = layer.arrk + layer.grad
            else:
                layer.weights=layer.weights+self.lr*layer.grad+self.momentum*layer.oldgrad
            '''elif layer.type=="conv2d":
                layer.arrk=layer.arrk+layer.grad'''


