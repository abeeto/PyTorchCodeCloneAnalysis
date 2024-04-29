import torch
from torch.autograd import Variable
import torch.nn as nn

use_cuda = torch.cuda.is_available()

if use_cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

x_data = Variable(torch.Tensor([[9.0], [16.0], [81.0]])).type(dtype)
y_data = Variable(torch.Tensor([[3.0], [4.0], [9.0]])).type(dtype)

class Model(torch.nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate a nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = nn.Linear(1, 50)  # One in and one out
        self.linear2 = nn.Linear(50, 100)  # One in and one out
        self.linear3 = nn.Linear(100, 1)  # One in and one out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out = self.linear(x)
        out = self.linear2(out)        
        y_pred = self.linear3(out)        
        return y_pred

# our model
model = Model()
model.cuda()
print(model)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

# Training loop
for epoch in range(5000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# After training
hour_var = Variable(torch.Tensor([[121.0]])).type(dtype)
y_pred = model(hour_var)
print("predict (after training)", 121, model(hour_var).data[0][0])