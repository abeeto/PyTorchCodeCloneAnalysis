import numpy as np
import random

class Tensor (object):

    def __init__(self, data, autograd=False, creators=None, creation_op=None, id=None):
        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None
        self.autograd = autograd
        self.children = {}
        if (id is None):
            id = random.randint(0, 100000)
        self.id = id

        ### Keep Track of how many children a tensor has
        if (creators is not None):
            for c in creators:
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1

    ### Check whether a tensor has received the correct number of gradients from each child
    def all_children_grads_accounted_for(self):
        for _,cnt in self.children.items():
            if(cnt != 0):
                return False
            return True

    ### Back Propogation
    def backward(self, grad=None, grad_origin=None):
        if(self.autograd):

            if(grad_origin is not None):

                if(self.children[grad_origin.id] == 0):
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1

            if(self.grad is None):
                self.grad = grad
            else:
                self.grad += grad               
 
            if(self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None)):
                
                ### Addition
                if(self.creation_op == "add"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                    self.grad += self.grad

                ### Negation
                if(self.creation_op == "neg"):
                    self.creators[0].backward(self.grad.__neg__())

                ### Subtraction
                    
                ### Multiplication

                ### Matrix Multiplication

                ### Transpose

                ### Summation

                ### Expansion

    def __add__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data + other.data, autograd=True, creators=[self,other], creation_op="add")
        return Tensor(self.data + other.data, creators = [self, other], creation_op="add")

    def __sub__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data - other.data, autograd=True, creators=[self, other], creation_op="sub")
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data * other.data, autograd=True, creators=[self, other], creation_op="mul")
        return Tensor(self.data * other.data)
    
    def sum(self, dim):
        if(self.autograd):
            return Tensor(self.data.sum(dim), autograd=True, creators=[self], creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)

        if (self.autograd):
            return Tensor(new_data, autograd=True, creators=[self], creation_op="expand_"+str(dim))
        return Tensor(new_data)

    def transpose(self):
        if(self.autograd):
            return Tensor(self.data.transpose(), autograd=True, creators=[self], creation_op="transpose")
        return Tensor(self.data.transpose())

    def mm(self, x):
        if(self.autograd):
            return Tensor(self.data.dot(x.data), autograd=True, creators=[self,x], creation_op="mm")
        return Tensor(self.data.dot(x.data))

    def __neg__(self):
        if(self.autograd):
            return Tensor(self.data * -1, autograd=True, creators=[self], creation_op="neg")
        return Tensor(self.data * -1)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())