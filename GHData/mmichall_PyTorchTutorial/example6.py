import torch
from torch.autograd import Variable  # for computational graphs
import torch.nn as nn  ## Neural Network package
import torch.optim as optim # Optimization package

# this block of code is organized a little differently than section 5, but it's mostly the same code
# the only three differences are:
# - The "Hyperparameter" constants
# - The for loop (for helping the model do <number of epochs> training steps)
# - The linear_layer1.zero_grad() function call on line 25.
#   (that's just to clear the gradients in memory, since we're starting the training over each iteration/epoch)

x1 = torch.Tensor([1, 2, 3, 4])
x1_var = Variable(x1, requires_grad=True)

linear_layer1 = nn.Linear(4, 1)

target_y = Variable(torch.Tensor([0]), requires_grad=False)

print("----------------------------------------")
print("Output (BEFORE UPDATE):")
print(linear_layer1(x1_var))

NUMBER_OF_EPOCHS = 3  # Number of times to update the weights
LEARNING_RATE = 1e-4  # Notice how I made the learning rate 1000 times smaller
loss_function = nn.MSELoss()
optimizer = optim.SGD(linear_layer1.parameters(), lr=LEARNING_RATE)

for epoch in range(NUMBER_OF_EPOCHS):
    linear_layer1.zero_grad()
    predicted_y = linear_layer1(x1_var)
    loss = loss_function(predicted_y, target_y)
    loss.backward()
    optimizer.step()

    print("----------------------------------------")
    print("Output (UPDATE " + str(epoch + 1) + "):")
    print(linear_layer1(x1_var))
    print("Should be getting closer to 0...")

print("----------------------------------------")

# here is where you might discover that training could take a *long* time
# we're barely doing anything, computationally speaking, and it's already scaling up
# in the next section, we're going to add more data (other than one sample with 4 features),
# and then, with each epoch, we're only going to use a small portion of it (called a "batch").