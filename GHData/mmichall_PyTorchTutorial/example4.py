import torch
from torch.autograd import Variable  # for computational graphs
import torch.nn as nn  ## Neural Network package

x1 = torch.Tensor([1, 2, 3, 4])
x1_var = Variable(x1, requires_grad=True)

linear_layer1 = nn.Linear(4, 1)

target_y = Variable(torch.Tensor([0]), requires_grad=False)
# ideally, we want our model to predict 0 when we input our x1_var variable below.
# here we're just sticking a Tensor with just 0 in it into a variable, and labeling it our target y value
# I put requires_grad=False because we're not computing any gradient with respect to our target (more on that later)

predicted_y = linear_layer1(x1_var)
print("----------------------------------------")
print(predicted_y)
print("----------------------------------------")
# prints 3.0995 for me; will probably be different for you.

loss_function = nn.MSELoss()
# this creates a function that takes a ground-truth Tensor and your model's output Tensor as inputs and calculates the "loss"
# in this case, it calculates the Mean Squared Error (a measurement for how far away your output is from where it should be)

loss = loss_function(predicted_y, target_y)
# here we actually use the function to compare our predicted_y vs our target_y

print(loss)
print("----------------------------------------")
# prints 9.6067 for me; will probably be different for you. It's just (target_y - predicted_y)^2 in this case.