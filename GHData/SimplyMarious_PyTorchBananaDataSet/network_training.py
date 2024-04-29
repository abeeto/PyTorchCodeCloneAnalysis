import torch
from torch.nn import Sequential, Linear, ReLU

# m = torch.nn.Linear(3,2)
# m.weight.data = torch.Tensor([[1,1,1], [2,2,2]])
# m.bias.data = torch.Tensor([1,1])
# x = torch.Tensor([1,2,3])
# y = m.forward(x)
#
# print("y: ", y)
#
# for p in m.parameters():
#     print(p)

#######################################################
#
# nn = Sequential(
#     Linear(3,2),
#     ReLU(),
#     Linear(2,2),
#     ReLU(),
#     Linear(2,2),
#     ReLU()
# )
#
# x = torch.Tensor([1,2,3])
# output = nn(x)
#
# print(f'input (shape: {x.shape}):\n{x}\noutput (shape: {output.shape}:\n{output}')
#
# print("parameters")
# for p in nn.parameters():
#     print(p)
#

#######################################################

# m = torch.nn.MSELoss()
# q = torch.Tensor([1.,2.,3.,4.])
# v = torch.Tensor([1.4,1.9,3.1,4.4])
#
# loss = m(q, v)
#
# print(loss)

#######################################################

from scipy.io import arff

with open('banana_dataset.arff') as dataset_file:
    data, meta = arff.loadarff(dataset_file)

X = []
Y = []
for x1, x2, c in data:
    x_elem = [x1, x2]
    label = torch.tensor(int(c)-1)
    X.append(x_elem)
    Y.append(label)

X = torch.tensor(X, dtype=torch.float)
Y = torch.tensor(Y, dtype=torch.long)

# plotting original data

# import matplotlib.pyplot as plotter
# plotter.figure(figsize=(10,10))
# plotter.scatter(X[Y==0,0], X[Y==0,1],
# edgecolors='g',facecolors='none',marker='o')
# plotter.scatter(X[Y==1,0], X[Y==1,1],
# edgecolors='y',facecolors='none',marker='o')
# plotter.show()

#################################################

from torch.nn import Sequential, Linear, ReLU

nn = Sequential(
    Linear(2,30),
    ReLU(),
    Linear(30, 2)
)

from torch.nn import CrossEntropyLoss
loss_object = CrossEntropyLoss()

from torch.optim import SGD
optimizer = SGD(nn.parameters(), lr=0.05)

nn.train(mode=True)

for epoch in range(0,1000):
    optimizer.zero_grad()
    outputs = nn(X)
    loss = loss_object(outputs, Y)
    loss.backward()
    optimizer.step()
    print(f'epoch:{epoch} loss:{loss}', end='\r')

nn.train(mode=False)
final_output = nn(X)
print(final_output)
predictions = torch.argmax(final_output, dim=1)

plotter.figure(figsize=(20,20))
plotter.scatter(X[Y==0,0], X[Y==0,1],
s=200,edgecolors='g',facecolors='none',marker='o')
plotter.scatter(X[Y==1,0], X[Y==1,1], s=200,
edgecolors='y',facecolors='none',marker='o')
plotter.scatter(X[predictions==0,0], X[predictions==0,1],facecolors='g',marker='x')
plotter.scatter(X[predictions==1,0], X[predictions==1,1],facecolors='y',marker='x')
print(f'accuracy score on training:{sum(Y==predictions)/len(Y)*100:.2f}%')
plotter.show()


