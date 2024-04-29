import collections
from os import read
import numpy as np

from core import TensorCore

'''
    Defines Tensor Object
'''

class Tensor(TensorCore):
    def __init__(
        self,
        data,
        creator: list = [],
        creation_op: str = None,
        requires_grad: bool = False,
        is_grad: bool = False,
        id = None,
        dtype = float
    ):
        super(Tensor, self).__init__(data, dtype)
        self.requires_grad = requires_grad
        self.grad = None
        self.creator = creator
        self.creation_op = creation_op
        self.is_grad = is_grad
        self.id = id if id is not None else np.random.randint(0, 1000000)
        self.children = collections.defaultdict(int)
        
        if not self.is_grad:
            for c in self.creator:
                c.children[self.id] += 1
    
    def _ready_to_bp(self) -> bool:
        return sum(self.children.values()) == 0

    def __add__(self, other):
        return Tensor(
            super().__add__(other),
            creator = [self, other],
            creation_op = 'add',
            requires_grad = self.requires_grad or other.requires_grad,
            is_grad = self.is_grad or other.is_grad
        )

    def __sub__(self, other):
        return Tensor(
            super().__sub__(other),
            creator = [self, other],
            creation_op = 'sub',
            requires_grad = self.requires_grad or other.requires_grad,
            is_grad = self.is_grad or other.is_grad
        )

    def __neg__(self):
        return Tensor(
            super().__neg__(),
            creator = [self],
            creation_op = 'neg',
            requires_grad = self.requires_grad,
            is_grad = self.is_grad
        )

    def __mul__(self, other):
        return Tensor(
            super().__mul__(other),
            creator = [self, other],
            creation_op = 'mul',
            requires_grad = self.requires_grad or other.requires_grad,
            is_grad = self.is_grad or other.is_grad
        )
    
    def mm(self, other):
        return Tensor(
            super().mm(other),
            creator = [self, other],
            creation_op = 'mm',
            requires_grad = self.requires_grad or other.requires_grad,
            is_grad = self.is_grad or other.is_grad
        )

    def transpose(self, axes = None):
        return Tensor(
            super().transpose(axes = axes),
            creator = [self],
            creation_op = 'transpose',
            requires_grad = self.requires_grad,
            is_grad = self.is_grad
        )

    def sum(self, axis = None):
        return Tensor(
            super().sum(axis = axis),
            creator = [self],
            creation_op = f'sum_{axis}',
            requires_grad = self.requires_grad,
            is_grad = self.is_grad
        )

    def expand_dims(self, axis):
        return Tensor(
            super().expand_dims(axis = axis),
            creator = [self],
            creation_op = f'expand_dims_{axis}',
            requires_grad = self.requires_grad,
            is_grad = self.is_grad
        )

    def repeat(self, repeat, axis):
        return Tensor(
            super().repeat(repeat = repeat, axis = axis),
            creator = [self],
            creation_op = f'repeat_{repeat}_{axis}',
            requires_grad = self.requires_grad,
            is_grad = self.is_grad
        )

    def expand(self, repeat, axis):
        return Tensor(
            super().expand(repeat = repeat, axis = axis),
            creator = [self],
            creation_op = f'expand_{repeat}_{axis}',
            requires_grad = self.requires_grad,
            is_grad = self.is_grad
        )

    def sigmoid(self):
        return Tensor(
            super().sigmoid(),
            creator = [self],
            creation_op = 'sigmoid',
            requires_grad = self.requires_grad,
            is_grad = self.is_grad
        )
    
    def tanh(self):
        return Tensor(
            super().tanh(),
            creator = [self],
            creation_op = 'tanh',
            requires_grad = self.requires_grad,
            is_grad = self.is_grad
        )


    def backward(self, grad_val, grad_origin = None):

        if not self.requires_grad:
            return
        
        if grad_origin is not None:
            if self.children[grad_origin.id] == 0:
                raise Exception("Cannot back propagate more than once")
            else:
                self.children[grad_origin.id] -= 1

        if self.grad is None:
            self.grad = grad_val
        else:
            self.grad += grad_val


        # Ready to BP
        if self.creator and (self._ready_to_bp() or grad_origin is None):
            if self.creation_op == "add":
                self.creator[0].backward(self.grad, self)
                self.creator[1].backward(self.grad, self)
            elif self.creation_op == "neg":
                self.creator[0].backward(self.grad.__neg__(), self)
            elif self.creation_op == "sub":
                self.creator[0].backward(self.grad, self)
                self.creator[1].backward(self.grad.__neg__(), self)
            elif self.creation_op == "mul":
                self.creator[0].backward(self.grad * self.creator[1], self)
                self.creator[1].backward(self.grad * self.creator[0], self)
            elif self.creation_op == "mm":
                self.creator[0].backward(self.grad.mm(self.creator[1].transpose()), self)
                self.creator[1].backward(self.creator[0].transpose().mm(self.grad), self)
            elif self.creation_op == "transpose":
                self.creator[0].backward(self.grad.transpose(), self)
            elif self.creation_op == "sigmoid":
                ones = Tensor(np.ones_like(self.grad.data))
                self.creator[0].backward(self.grad * self * (ones - self), self)
            elif self.creation_op == "tanh":
                ones = Tensor(np.ones_like(self.grad.data))
                self.creator[0].backward(self.grad * (ones - self * self), self)
            elif self.creation_op.startswith('sum'):
                axis = int(self.creation_op.split('_')[1])
                repeat = self.creator[0].shape()[axis]
                self.creator[0].backward(self.grad.expand_dims(axis).repeat(repeat, axis), self)
            elif self.creation_op.startswith('repeat'):
                axis = int(self.creation_op.split("_")[1])
                self.creator[0].backward(self.grad.sum(axis), self)
            elif self.creation_op.startswith('expand'):
                repeat = int(self.creation_op.split("_")[1])
                axis = int(self.creation_op.split("_")[2])
                self.creator[0].backward(self.grad.sum(axis))



if __name__ == '__main__':

    X_xor = [
        [1, 1],
        [0, 1],
        [1, 0],
        [0, 0]
    ]

    testTensor = Tensor(X_xor)

    print('Pause')
