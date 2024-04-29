#############################################
# ------------------------------------------ #
# Training One Parameter
# ------------------------------------------ #
#############################################
# These are the libraries will be used for this lab.
import torch
import numpy as np
import matplotlib.pyplot as plt


# The class for plotting
class plot_diagram():
    # Constructor
    def __init__(self, X, Y, w, stop, go=False):
        start = w.data
        self.error = []
        self.parameter = []
        self.X = X.numpy()
        self.Y = Y.numpy()
        self.parameter_values = torch.arange(start, stop)
        self.Loss_function = [criterion(forward(X), Y) for w.data in self.parameter_values]
        w.data = start

    # Executor
    def __call__(self, Yhat, w, error, n):
        self.error.append(error)
        self.parameter.append(w.data)
        plt.subplot(212)
        plt.plot(self.X, Yhat.detach().numpy())
        plt.plot(self.X, self.Y, 'ro')
        plt.xlabel("A")
        plt.ylim(-20, 20)
        plt.subplot(211)
        plt.title("Data Space (top) Estimated Line (bottom) Iteration " + str(n))
        plt.plot(self.parameter_values.numpy(), self.Loss_function)
        plt.plot(self.parameter, self.error, 'ro')
        plt.xlabel("B")
        plt.figure()
        plt.show()

    # Destructor
    def __del__(self):
        plt.close('all')


# Create the f(X) with a slope of -3
# view(-1, 1): reshape the tensor to a column vector
X = torch.arange(-3, 3, 0.1).view(-1, 1)
f = -3 * X

# Add some Gaussian noise to f(X) and save it in Y
Y = f + 0.1 * torch.randn(X.size())


# Create forward function for prediction
def forward(x):
    return w * x


# Create the MSE function for evaluate the result.
def criterion(yhat, y):
    return torch.mean((yhat - y) ** 2)


# Create Learning Rate and an empty list to record the loss for each iteration
lr = 0.1
LOSS = []
w = torch.tensor(-10.0, requires_grad=True)
gradient_plot = plot_diagram(X, Y, w, stop=5)


# Define a function for train the model
def train_model(iter):
    for epoch in range(iter):
        # make the prediction as we learned in the last lab
        Yhat = forward(X)

        # calculate the iteration
        loss = criterion(Yhat, Y)

        # plot the diagram for us to have a better idea
        gradient_plot(Yhat, w, loss.item(), epoch)

        # store the loss into list
        LOSS.append(loss.item())

        # backward pass: compute gradient of the loss with respect to all the learnable parameters
        loss.backward()
        # print("Gradient data:", w.grad.data)

        # Update parameters
        # print("Before update: ", w.data)
        # w.data = w.data - lr * w.grad.data
        # print("After update: ", w.data)

        # w.grad.data = w.grad: both of them can retrieve a tensor type of data
        # print(w.grad.data)
        # print(w.grad)

        # zero the gradients before running the backward pass
        # If we don't zero the gradient, it will accumulate instead of replacing
        # loss.backward() actually adds dy/dx to the current value of x.grad
        # x.grad += true_gradient
        # single_training_underscore_
        # This convention could be used for avoiding conflict
        # with Python keywords or built-ins.
        # print("Before zero out: ", w.grad.data)
        w.grad.data.zero_()
        # print("after zero out: ", w.grad.data)
        # print("Without zeroing out: ", w.grad.data)


# Give 4 iterations for training the model here.
train_model(4)
plt.plot(LOSS)
plt.tight_layout()
plt.xlabel("Epoch/Iterations")
plt.ylabel("Cost")
plt.show()
