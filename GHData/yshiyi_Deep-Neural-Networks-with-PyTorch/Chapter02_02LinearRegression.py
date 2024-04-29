# These are the libraries will be used for this lab.
import torch

# # ------------------------------------------ #
# # Simple Linear Regression: Prediction
# # ------------------------------------------ #
# # Define w = 2 and b = -1 for y = wx + b
# # Variables defined outside are global variables.
# # They can be used in any functions, but they can't be modified.
# # To change they inside a function, must use global var_name
# w = torch.tensor(2.0, requires_grad=True)
# b = torch.tensor(-1.0, requires_grad=True)
#
#
# # Function forward(x) for prediction
# def forward(x):
#     yhat = w * x + b
#     return yhat
#
#
# # Predict y = 2x - 1 at x = 1
# x = torch.tensor([[1.0]])
# yhat = forward(x)
# print("The prediction: ", yhat)
# # Create x Tensor and check the shape of x tensor
# x = torch.tensor([[1.0], [2.0]])
# print("The shape of x: ", x.shape)
# yhat = forward(x)
# print("The prediction: ", yhat)
#
#
# ----------------------------------------------------- #
# Class Linear
# The linear class can be used to make a prediction.
# Note: If we create the mode using torch.nn.Linear,
#       then the keys are 'weight' and 'bias'.
# ----------------------------------------------------- #
# Import Class Linear
from torch import nn
# from torch.nn import Linear  # This also works
# Set random seed
torch.manual_seed(1)

# Create Linear Regression Model, and print out the parameters
# in_features: dim of input; out_features: dim of output
# bias=True: there is b; False: no b
lr = nn.Linear(in_features=1, out_features=1, bias=True)
print("Parameters w and b: ", list(lr.parameters()))
# Parameters are contained in a dictionary with keys "weight" and "bias"
print("Python dictionary: ", lr.state_dict())
print("keys: ", lr.state_dict().keys())
print("values: ", lr.state_dict().values())
print("weight:", lr.weight)  # lr.weight[0][0].item()
print("bias:", lr.bias)
x = torch.tensor([1.0])
yhat = lr(x)
print("The prediction: ", yhat)


# ----------------------------------------------------- #
# Build Custom Modules
# Note: If we create the mode using custom module,
#       then the keys are 'linear.weight' and 'linear.bias'.
# ----------------------------------------------------- #
# Library for this section
from torch import nn


# Customize Linear Regression Class
class LR(nn.Module):
    # Constructor
    def __init__(self, input_size, output_size):
        # Inherit from parent
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    # Prediction function
    # This's the forward function that defines the network structure
    # It works like __call__
    def forward(self, x):
        out = self.linear(x)
        return out


lr = LR(1, 1)
print("The parameters: ", list(lr.parameters()))  # lr.linear.weight, lr.linear.bias
print("Linear model: ", lr.linear)
print("Python dictionary: ", lr.state_dict())  # lr.linear.state_dict() also works
print("keys: ", lr.state_dict().keys())
print("values: ", lr.state_dict().values())
print("Value of first key:", lr.state_dict()["linear.weight"][0].item())


x = torch.tensor([[2.0]])
yhat = lr(x)
print("The prediction: ", yhat)
