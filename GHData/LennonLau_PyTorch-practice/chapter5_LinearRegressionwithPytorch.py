import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
lr = 0.01
epochs = 1000


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat


model = LinearModel()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


epoch_list = []
loss_list = []
for epoch in range(epochs):
    y_hat = model(x_data)  # mini_batch = 3
    loss = criterion(y_hat, y_data)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print('epoch:{}  w:{}  b:{}  loss:{}'.format(epoch+1, model.linear.weight.item(),
                                                 model.linear.bias.item(), loss.item()))
    epoch_list.append(epoch)
    loss_list.append(loss.item())

plt.plot(epoch_list, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()