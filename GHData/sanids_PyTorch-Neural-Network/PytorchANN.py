#simple neural network in Pytorch
from __future__ import print_function, division
from builtins import range



import numpy as np
import matplotlib.pyplot as plt
from util import get_normalized_data

import torch
from torch.autograd import Variable
from torch import optim




model = torch.nn.Sequential()


# ANN with layers [784] -> [500] -> [300] -> [10]
model.add_module("dense1", torch.nn.Linear(D, 500))
model.add_module("relu1", torch.nn.ReLU())
model.add_module("dense2", torch.nn.Linear(500, 300))
model.add_module("relu2", torch.nn.ReLU())
model.add_module("dense3", torch.nn.Linear(300, K))
# Note: no final softmax!
# just like Tensorflow, it's included in cross-entropy function


# define a loss function

loss = torch.nn.CrossEntropyLoss(size_average=True)



# define an optimizer

optimizer = optim.Adam(model.parameters(), lr=1e-4)




def train(model, loss, optimizer, inputs, labels):
  inputs = Variable(inputs, requires_grad=False)
  labels = Variable(labels, requires_grad=False)

  # Reset gradient
  optimizer.zero_grad()

  # Forward
  logits = model.forward(inputs)
  output = loss.forward(logits, labels)

  # Backward
  output.backward()

  # Update parameters
  optimizer.step()

  # what's the difference between backward() and step()?

  return output.item()


# similar to train() but not doing the backprop step
def get_cost(model, loss, inputs, labels):
  inputs = Variable(inputs, requires_grad=False)
  labels = Variable(labels, requires_grad=False)

  # Forward
  logits = model.forward(inputs)
  output = loss.forward(logits, labels)

  return output.item()


# define the prediction procedure
# also encapsulate these steps
# Note: inputs is a torch tensor
def predict(model, inputs):
  inputs = Variable(inputs, requires_grad=False)
  logits = model.forward(inputs)
  return logits.data.numpy().argmax(axis=1)


# return the accuracy
# labels is a torch tensor
# to get back the internal numpy data
# use the instance method .numpy()
def score(model, inputs, labels):
  predictions = predict(model, inputs)
  return np.mean(labels.numpy() == predictions)


### prepare for training loop ###

# convert the data arrays into torch tensors
Xtrain = torch.from_numpy(Xtrain).float()
Ytrain = torch.from_numpy(Ytrain).long()
Xtest = torch.from_numpy(Xtest).float()
Ytest = torch.from_numpy(Ytest).long()

# training parameters
epochs = 15
batch_size = 32
n_batches = Xtrain.size()[0] // batch_size

# things to keep track of
train_costs = []
test_costs = []
train_accuracies = []
test_accuracies = []

# main training loop
for i in range(epochs):
  cost = 0
  test_cost = 0
  for j in range(n_batches):
    Xbatch = Xtrain[j*batch_size:(j+1)*batch_size]
    Ybatch = Ytrain[j*batch_size:(j+1)*batch_size]
    cost += train(model, loss, optimizer, Xbatch, Ybatch)

  
  # we could have also calculated the train cost here
  # but I wanted to show you that we could also return it
  # from the train function itself
  train_acc = score(model, Xtrain, Ytrain)
  test_acc = score(model, Xtest, Ytest)
  test_cost = get_cost(model, loss, Xtest, Ytest)

  print("Epoch: %d, cost: %f, acc: %.2f" % (i, test_cost, test_acc))

  # for plotting
  train_costs.append(cost / n_batches)
  train_accuracies.append(train_acc)
  test_costs.append(test_cost)
  test_accuracies.append(test_acc)



# plot the results
plt.plot(train_costs, label='Train cost')
plt.plot(test_costs, label='Test cost')
plt.title('Cost')
plt.legend()
plt.show()

plt.plot(train_accuracies, label='Train accuracy')
plt.plot(test_accuracies, label='Test accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()