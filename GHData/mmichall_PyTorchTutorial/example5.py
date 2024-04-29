import torch
from torch.autograd import Variable  # for computational graphs
import torch.nn as nn  ## Neural Network package
import torch.optim as optim # Optimization package

# Now, instead of calculating the gradient of our linear layer wrt our inputs (x) in lesson 3,
# we're going to calculate the gradient of our loss function wrt our weights / biases

x1 = torch.Tensor([1, 2, 3, 4])
x1_var = Variable(x1, requires_grad=True)

linear_layer1 = nn.Linear(4, 1)

target_y = Variable(torch.Tensor([0]), requires_grad=False)

predicted_y = linear_layer1(x1_var)

loss_function = nn.MSELoss()

loss = loss_function(predicted_y, target_y)

optimizer = optim.SGD(linear_layer1.parameters(), lr=1e-1)
# here we've created an optimizer object that's responsible for changing the weights
# we told it which weights to change (those of our linear_layer1 model) and how much to change them (learning rate / lr)
# but we haven't quite told it to change anything yet. First we have to calculate the gradient.

loss.backward()

# now that we have the gradient, let's look at our weights before we change them:

print("----------------------------------------")
print("Weights (before update):")
print(linear_layer1.weight)
print(linear_layer1.bias)
# let's also look at what our model predicts the output to be:

print("----------------------------------------")
print("Output (before update):")
print(linear_layer1(x1_var))

optimizer.step()
# we told the optimizer to subtract the learning rate * the gradient from our model weights

print("----------------------------------------")
print("Weights (after update):")
print(linear_layer1.weight)
print(linear_layer1.bias)

# looks like our weights and biases changed. How do we know they changed for the better?
# let's also look at what our model predicts the output to be now:

print("----------------------------------------")
print("Output (after update):")
print(linear_layer1(x1_var))
print("----------------------------------------")

# wow, that's a huge change (at least for me, and probably for you). It looks like our learning rate might be too high.
# perhaps we want to make our model learn slower, compensating with more than one weight update?
# next section!