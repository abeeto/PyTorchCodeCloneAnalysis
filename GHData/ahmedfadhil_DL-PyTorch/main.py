import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Create a torch tensor
# a = torch.Tensor([1, 2, 3, 4, 5, 5, 4, 3, 2, 3])
# a.size()
#
# b = torch.Tensor([[1, 2, 3, 4, 5, 6, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6, 4, 3, 2, 1]])
# b.mean()
# b.size()
#
# a = Variable(torch.ones(2, 2), requires_grad=True)
#
# b = Variable(torch.ones(2, 2), requires_grad=True)
#
# print(torch.add(a, b))
#
# # Gradients
#
# x = Variable(torch.ones(2), requires_grad=True)
#
# y = 5*(x+1)**2
# print(y)
# o = (1/2)*torch.sum(y)
#
#
# o.backward()
# x.grad
#
# o.backward(torch.FloatTensor([1.0,1.0]))
#
#

# np.random.seed(1)
# n = 50
# x = np.random.randn(n)
# y = x * np.random.randn(n)
#
# colors = np.random.randn(n)
# plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
# plt.sca(x, y, c=colors, alpha=0.5)
# plt.show()

# CPU implementation
x_values = [i for i in range(11)]

# Convert to numpy
x_train = np.array(x_values, dtype=np.float32)
x_train.shape

x_train = x_train.reshape(-1, 1)
x_train.shape

y_values = [2 * i + 1 for i in x_values]
# y_values = []
# for i in x_values:
#     result = 2 * i + 1
#     y_values.append(result)
y_train = np.array(y_values, dtype=np.float32)

y_train = y_train.reshape(-1, 1)
y_train.shape

# Building the model
import torch.nn as nn
from torch.autograd import Variable


# Create the class
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)

# For GPU model
if torch.cuda.is_available():
    model.cuda()

# Instantiate Loss Class
criterion = nn.MSELoss()

# Instantiate Optimizer Class
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Model training
epochs = 100
for epoch in range(epochs):
    # Convert numpy array to torch Variable
    # For GPU
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))

    # Clear gradients wrt parameters
    optimizer.zero_grad()

    # Forward to get output
    outputs = model(inputs)

    #     Calculate loss
    loss = criterion(outputs, labels)

    #     Getting gradients wrt parameters
    loss.bacward()

    #     Update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epochs, loss.data[0]))

#     Purely inference
predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()

print(y_train)

# Clear figure
plt.clf()

# Get predictions
predicted = model(Variable(torch.from_numpy(x_train))).data.train()

# Plot true data
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)

# Plot predictions
plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)

# Legend and plot
plt.legend(loc='best')
plt.show()

# Save the model
save_model = False
# if save_model == True:
if save_model is True:
    #      Save only parameters
    # alph and beta
    torch.save(model.state_dict(), 'awesome_model.pkl')

#     Load the model

load_model = False
if load_model == True:
    model.load_state_dict(torch.load('awesome_model.pkl'))

#     GPU implementation


x = [1, 2, 3]
y = [1, 2, 3]
colors = np.random.rand(len(x))
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
plt.xlabel("some text")
plt.ylabel("some text")

plt.scatter(x, y, c=colors, alpha=0.5)
plt.show()


