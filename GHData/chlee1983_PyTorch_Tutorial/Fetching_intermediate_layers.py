import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt

x = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [[3], [7], [11], [15]]

X = torch.tensor(x).float()
Y = torch.tensor(y).float()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
X = X.to(device)
Y = Y.to(device)


class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(2, 8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8, 1)

    def forward(self, x):
        hidden1 = self.input_to_hidden_layer(x)
        hidden2 = self.hidden_layer_activation(hidden1)
        x = self.hidden_to_output_layer(hidden2)
        return x, hidden1


torch.random.manual_seed(10)
mynet = MyNeuralNet().to(device)

loss_func = nn.MSELoss()

_Y, _Y_hidden = mynet(X)
loss_value = loss_func(_Y, Y)
opt = SGD(mynet.parameters(), lr = 0.001)

loss_history = []
for _ in range(50):
    opt.zero_grad()
    loss_value = loss_func(mynet(X)[0], Y)
    loss_value.backward()
    opt.step()
    loss_history.append(loss_value.detach())

plt.plot(loss_history)
plt.title("Loss Variation Over Increasing Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")
plt.show()

mynet_output = mynet(X)[0]
print(mynet_output)

mynet_post_activation = mynet(X)[1]
print(mynet_post_activation)