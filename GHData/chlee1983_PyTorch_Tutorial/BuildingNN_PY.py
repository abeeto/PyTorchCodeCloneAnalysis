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
        '''used for illustration of Linear method'''
        # print(nn.Linear(2, 7))

    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x


mynet = MyNeuralNet().to(device)

'''used for illustration of how to obtain parameters of a given layer'''
# print(mynet.hidden_to_output_layer.weight)

'''used for illustration of how to obtain parameters of all layers in a model'''
# mynet.parameters()

'''used for illustration of how to obtain parameters of all layers in a model by looping through the generator object'''
# for par in mynet.parameters():
#     print(par)

loss_func = nn.MSELoss()

# prediction of neural network
_Y = mynet(X)

'''PyTorch convention to send prediction value first then the ground truth'''
# loss_value = loss_func(_Y, Y)

# optimizer that tries to reduce the loss value
opt = SGD(mynet.parameters(), lr=0.001)


loss_history = []
for _ in range(50):
    # flush the previous epoch's gradients
    opt.zero_grad()
    # compute loss
    loss_value = loss_func(mynet(X), Y)
    # perform back-propagation
    loss_value.backward()
    # update the weights according to the gradients computed
    opt.step()
    # storing loss value in each epoch in the list
    loss_history.append(loss_value.detach())


plt.plot(loss_history)
plt.title('Loss variation over increasing epochs')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.show()
