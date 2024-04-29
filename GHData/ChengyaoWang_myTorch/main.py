from tensor import Tensor
from optimizer import SGD
from layer import MSELoss, Linear, Tanh, Sigmoid
from model import Sequential

import numpy as np

#Toy example of Using Tensor Class
np.random.seed(0)
data = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), requires_grad = True)
target = Tensor(np.array([[0], [1], [0], [1]]), requires_grad = True)
#Every element in w, is an Object of Tensor representing weight matrix
model = Sequential(
    Linear(2,3),
    Tanh(),
    Linear(3,3),
    Tanh(),
    Linear(3,1),
)
optim = SGD(parameters = model.get_parameters(), lr = 0.1)
criterion = MSELoss()
for i in range(10):
    pred = model(data)
    loss = criterion(pred, target)
    loss.backward(Tensor(np.ones_like(loss.data), is_grad = True))
    optim.step()
    print(loss.data)
print("------------------------------------------------------------------------")