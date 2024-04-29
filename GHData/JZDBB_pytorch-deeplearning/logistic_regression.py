import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([1.0], [2.0], [3.0]))
y_data = Variable(torch.Tensor([2.0], [4.0], [6.0]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred
