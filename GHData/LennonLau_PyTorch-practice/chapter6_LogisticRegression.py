import torch
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --------------------------------------------------------------------------------- #


# prepare the train_set
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])
# --------------------------------------------------------------------------------- #


# super parameters
lr = 0.02
epochs = 1000
# --------------------------------------------------------------------------------- #


# define the network
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        x = torch.nn.functional.sigmoid(self.linear(x))
        return x


model = LogisticRegression()
# --------------------------------------------------------------------------------- #


# define the loss_function and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# --------------------------------------------------------------------------------- #


# the training cycle
epoch_list = []
loss_list = []
for epoch in range(epochs):
    y_hat = model(x_data)
    loss = criterion(y_hat, y_data)
    loss.backward()
    optimizer.step()
    print('epoch={}, loss={}, w={}, b={}'.format(epoch+1, loss.item(), model.linear.weight.item(), model.linear.bias.item()))
    optimizer.zero_grad()
    epoch_list.append(epoch)
    loss_list.append(loss.item())

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
# --------------------------------------------------------------------------------- #


# use the test_set to predict
x_test = np.linspace(0, 10, 200)
x_test = torch.Tensor(x_test).view((200, 1))
y_pred = model(x_test)
y_pred = y_pred.data.numpy()
plt.plot(x_test, y_pred)
plt.xlabel('x_test')
plt.ylabel('y_pres')
plt.show()