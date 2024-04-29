import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


device = 'cpu'

file = open("regres_input.txt", "r")
x_i = [i for i in range (1,26)]
y_i = [float(i) for i in file.read().splitlines()]
file.close()
x_i, y_i = np.array(x_i), np.array(y_i)
x_i_norm = (x_i - x_i.mean())/np.sqrt(x_i.var())

X_train, X_test, y_train, y_test = train_test_split(x_i_norm, y_i, train_size=0.8, random_state=1)
X_train_old = X_train
X_train, X_test = X_train.reshape(-1,1), X_test.reshape(-1,1)
y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)


#___Defining the model___
my_model = nn.Sequential(
    nn.Linear(1, 4),
    nn.LeakyReLU(),
    nn.Linear(4, 1)
)
#my_model = model().to(device)
print(my_model.state_dict())

torch.manual_seed(51)
learning_rate = 0.05
loss_fn = nn.MSELoss()
optimizer = optim.Adamax(my_model.parameters(), lr=learning_rate)
#optimizer = optim.Adam(my_model.parameters(), lr=learning_rate)
EPOCHS = 1000

inputs = Variable(torch.from_numpy(X_train).float())


for epoch in range(EPOCHS): 
    my_model.train()

    labels = Variable(torch.from_numpy(y_train).float())
    preds = my_model(inputs)

    loss = loss_fn(preds, labels)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    train_loss = loss.item()

    #validation
    my_model.eval()

    with torch.no_grad():
        target = Variable(torch.from_numpy(y_test).float())
        outp = my_model(Variable(torch.from_numpy(X_test).float()))
        loss = loss_fn(target, outp)
        valid_loss = loss.item()
    

    if (epoch % 10 == 0):
        print(f'epoch: {epoch}, Train_Loss: {train_loss}, Valid_Loss: {valid_loss}')


with torch.no_grad():
    preds = my_model(Variable(torch.from_numpy(X_train).float())).data.numpy()

plt.grid(which='major', color = 'k')
plt.grid(which='minor', color = 'grey', linestyle = ':')
graph1 = plt.scatter(x_i_norm, y_i, color='blue', label='Sample')
graph2 = plt.plot(X_train_old, preds, color='magenta', label='Prediction')
plt.show()