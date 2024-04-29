##########################################################
# Linear Classifier: Logistic Regression
##########################################################
import torch
import matplotlib.pyplot as plt
torch.manual_seed(1)


z = torch.arange(-100, 100, 0.1).view(-1, 1)
# Create a sigmoid object
sig = torch.nn.Sigmoid()
yhat = sig(z)
plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')
# plt.show()

yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())
# plt.show()


##########################################################
# Build a Logistic Regression with nn.Sequential
##########################################################
x = torch.tensor([[1.0]])
X = torch.tensor([[1.0], [100]])
model = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Sigmoid())
# Print the parameters
print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())

# The prediction for x
yhat = model(x)
print("The prediction: ", yhat)
# The prediction for X
yhat = model(X)
print("The prediction: ", yhat)


##########################################################
# Multiple features
##########################################################
x = torch.tensor([[1.0, 1.0]])
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])

model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())
# Make the prediction of x
yhat = model(x)
print("The prediction: ", yhat)
# The prediction of X
yhat = model(X)
print("The prediction: ", yhat)


##########################################################
# Custom Module for Logistic Regression
##########################################################
class logtistic_regression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        yhat = torch.sigmoid(self.linear(x))
        return yhat


# Predict with one feature
x = torch.tensor([[1.0]])
X = torch.tensor([[-100], [0], [100.0]])
model = logtistic_regression(1, 1)
# Make the prediction of x
yhat = model(x)
print("The prediction: ", yhat)
# The prediction of X
yhat = model(X)
print("The prediction: ", yhat)

# Predict with two features
x = torch.tensor([[1.0, 2.0]])
X = torch.tensor([[100, -100], [0.0, 0.0], [-100, 100]])
model = logtistic_regression(2, 1)
# Make the prediction of x
yhat = model(x)
print("The prediction: ", yhat)
# The prediction of X
yhat = model(X)
print("The prediction: ", yhat)
