# This code shows how to combine softmax and cross entropy.
# for a loss function, we can just input the Y, and Y_predict, and it will compute us the loss, Y and Y_predict
# can be vector, or the collection of multiple training example. Softmax combined with corss entropy is just used 
# for training, for prediction, we just find the maximum neuron. Like the binary classification, sigmoid + binary 
# cross entropy is just used for training, for prediction, we can just see if the neuron is >0 or <0, or after sigmoid, 
# > or < than 0.5.

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np 

# compute loss of a one-hot vector
Y = np.array([1,0,0]) # one hot, indicate index 0

Y_pred1 = np.array([0.7, 0.2, 0.1])
Y_pred2 = np.array([0.1, 0.3, 0.6]) # two one hot predictions

print("loss1 = ", np.sum(-Y*np.log(Y_pred1)))
print("loss2 = ", np.sum(-Y*np.log(Y_pred2)))


# There are three ways of indicating class N: 1. one 0-(N-1) index, 2. one N dimensional one-hot normalized vector, 3. one N dimensional vector, but not normalized as one-hot, but there is one maxima.
# We can compute the loss using the correct Y, as a class index, and raw prediction, which may not be soft-max normalized
loss = nn.CrossEntropyLoss() # Like a criterion

Y = torch.tensor([0]) # the correct target, as in one class index

Y_pred1 = torch.tensor([[2.0, 1.0, 0.1]]) # predict vector, before softmax normalization
Y_pred2 = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print("\nLoss1 = ", l1.data, "\nLoss2 = ", l2.data) # print the loss for two predictions,

print("Y_pred1=", torch.max(Y_pred1.data,1)[1]) # from the raw prediction, we need to find the class index it means, torch.max(..., 0), is to take max vertically, torch.max(...,1) is horizontally, after torch.max, there are two index, [0] is the maximum value, [1] is the index of this maximum value.
print("Y_pred2=", torch.max(Y_pred2.data,1)[1])



# For a batch, or the totally traning set, there are multiple training examples, we need to import them together
Y = torch.tensor([2,0,1], requires_grad = False) # three training examples, with class 2,0,1

Y_pred1 = torch.tensor([[1.0, 2.0, 9.0], [1.1, 0.1, 0.2], [0.2, 2.1, 0.1]]) #Batch 1 has three examples, three predictions, each prediction has three energies
Y_pred2 = torch.tensor([[0.8, 0.2, 0.3], [0.2, 0.3, 0.5], [0.2, 0.2, 0.5]])

l1 = loss(Y_pred1, Y) # the loss for the whole batch
l2 = loss(Y_pred2, Y)

print("\nBatch loss1 =", l1.data, "\n Batch loss2 =", l2.data)
