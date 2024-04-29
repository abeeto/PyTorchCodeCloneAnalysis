from Module import Module 

class Optim():
    def __init__(self, module, lr=0.01):
        self.module = module
        self.lr = lr 

        self.parameters = []
        for layer in module.layers:
            for param in layer.parameters:
                self.parameters.append(param)
            
    def get_vars(self, curr_var, vars):
        if curr_var is not None:
            for var in curr_var.fn_inputs:
                vars.append(var)
                self.get_vars(var, vars)


    def zero_grad(self):
        vars = []
        self.get_vars(self.module.output, vars)
        for var in vars:
            var.grad = 0

    def step(self):
        for param in self.parameters:
            param.data -= self.lr*param.grad