import torch
from torch.autograd import Variable # for computational graphs
import torch.nn as nn ## Neural Network package

x1 = torch.Tensor([1, 2, 3, 4])
x1_var = Variable(x1, requires_grad=True)

linear_layer1 = nn.Linear(4, 1)
# create a linear layer (i.e. a linear equation: w1x1 + w2x2 + w3x3 + w4x4 + b, with 4 inputs and 1 output)
# w and b stand for weight and bias, respectively

predicted_y = linear_layer1(x1_var)
# run the x1 variable through the linear equation and put the output in predicted_y

print("----------------------------------------")
print(predicted_y)
print("----------------------------------------")
# prints the predicted y value (the weights and bias are initialized randomly; my output was 1.3712)