import torch
from torch.autograd import Variable  # for computational graphs
import torch.nn as nn  ## Neural Network package

x1 = torch.Tensor([1, 2, 3, 4])
x1_var = Variable(x1, requires_grad=True)

linear_layer1 = nn.Linear(4, 1)

target_y = Variable(torch.Tensor([0]), requires_grad=False)

predicted_y = linear_layer1(x1_var)

# at this point, we want the gradient of our linear layer with respect to our original input, x
# the Variable object we put our Tensor in is supposed to store its respective gradients, so let's look:

print("----------------------------------------")
print(x1_var.grad)
print("----------------------------------------")

# this prints None, because we haven't computed any gradients yet.
# we have to call the backward() function from our predicted results in order to compute gradients with respect to x

predicted_y.backward()
print(x1_var.grad)
print("----------------------------------------")
# This is the gradient Tensor that holds the partial derivatives of our linear function with respect to each entry in x1