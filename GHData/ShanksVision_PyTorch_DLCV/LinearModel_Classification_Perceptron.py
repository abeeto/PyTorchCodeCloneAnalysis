# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:09:16 2020

@author: shankarj
"""
import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dt

class LogisticReg(pt.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.logMod = pt.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        pred = pt.sigmoid(self.logMod(x))
        return pred
    
    def predict(self, x):
        val = self.forward(x)
        return pt.round(val)
    
    def get_params(self):
        [w, b] = self.logMod.parameters()
        w1, w2 = w.view(2)
        return (w1.item(), w2.item(), b[0].item())
    
#Create random data and fit the model
centers = [[-0.5, -0.5], [0.5, 0.5]]
data_pts = 300
x, y = dt.make_blobs(data_pts, 2, centers, 0.5)

x = pt.tensor(x, dtype=pt.float32)
y = pt.tensor(y, dtype=pt.float32).view(300,1)

lr = LogisticReg(2, 1)
#before the fit
w1, w2, b1 = lr.get_params()
x1 = np.array([pt.min(x, 0)[0][0], pt.max(x, 0)[0][0]])
x2 = (w1*x1 + b1)/(-w2)

#model compilation
objective = pt.nn.BCELoss()
optimizer = pt.optim.SGD(lr.parameters(), lr=0.03)
epochs = 500
loss_history =[]

#model fit
for i in range(epochs):
    y_pred = lr.forward(x)
    loss = objective(y_pred, y)
    loss_history.append(loss)
    print(f'Epoch {i+1} : Loss value {loss:.5f}')
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#before the fit
w1, w2, b1 = lr.get_params()
x1_final = np.array([pt.min(x, 0)[0][0], pt.max(x, 0)[0][0]])
x2_final = (w1*x1 + b1)/(-w2)    

y_vec = y.view(-1)
plt.scatter(x.numpy()[y_vec==0, 0], x.numpy()[y_vec==0, 1], label='class 0')
plt.scatter(x.numpy()[y_vec==1, 0], x.numpy()[y_vec==1, 1], label='class 1')
plt.plot(x1, x2, 'g-', label='Initial classifier')
plt.plot(x1_final, x2_final, 'y', label='final classifier')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#test model
test_vals = pt.tensor([[0., 1.], [0., -1.]], dtype=pt.float32)

print(f'data point {test_vals[0]} will be in class {lr.predict(test_vals[0]).item()}')
print(f'data point {test_vals[1]} will be in class {lr.predict(test_vals[1]).item()}')
