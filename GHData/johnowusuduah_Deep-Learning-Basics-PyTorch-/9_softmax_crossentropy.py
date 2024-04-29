import torch
import torch.nn as nn
import numpy as np


# i. SOFTMAX FROM SCRATCH ---> NUMPY ARRAYS
#*******************************************************************************
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print("softmax numpy:", outputs)


# ii. SOFTMAX FROM SCRATCH ---> TORCH TENSORS
#*******************************************************************************
x = torch.tensor([2.0, 1.0, 0.1], dtype=torch.float32)
# dim = 0 adds values along rows
outputs = torch.softmax(x, dim=0)
print(outputs)


# iii. CROSS ENTROPY LOSS ---> TORCH TENSORS
#*******************************************************************************
# WORDS OF CAUTION
#*******************************************************************************
# nn.CrossEntropyLoss applies nn.LogSoftmax and nn.LLLoss (Negative Log Likelihood Loss)

# we should not implement Softmax in the last layer

# Y has class labels, not One-Hot Encoded Labels

# Y_pred has raw scores (logits), not softmax

# PyTorch Cross Entropy
loss = nn.CrossEntropyLoss()

# ONE SAMPLE

# A. DEFINITION OF DATA
# y is a tensor of labels not one-hot encoded
Y = torch.tensor([0])
# nsamples x nclasses = 1 x 3
# y_pred are logit scores not probabilities
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.1, 0.3, 0.6]])

# B. LOSS COMPUTATION
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f"Loss of good tensor: {l1.item()}")
print(f"Loss of bad tensor: {l2.item()}")


# B. PREDICTIONS
print(Y_pred_good.shape)
print(torch.max(Y_pred_good, 1))

# 1 means across the first dimension (i.e. rows) aka compare values in each row
_, predicitons1 = torch.max(Y_pred_good, 1)
_, predicitons2 = torch.max(Y_pred_bad, 1)


# MULTIPLE SAMPLES

# A. DEFINITION OF DATA
Y = torch.tensor([2, 0, 1])

# nsamples x nclasses = 3 x 3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())


# iv. APPLICATION OF CROSSENTROPY LOSS IN MULTICLASS CLASSIFICATION NEURAL NETWORK
#**********************************************************************************
# For multiclass problem
class NeuralNet2(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet2, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    out = self.linear1(x)
    out = self.relu(out)
    out = self.linear2(out)
    # no softmax at the end
    return out

  # OR
  def forward(self, x):
    out = torch.relu(self.linear1(x))
    out = self.linear2(self.linear2(out))
    return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss() #(applies softmax)


# v. APPLICATION OF CROSSENTROPY LOSS IN BINARY CLASSIFICATION NEURAL NETWORK
#**********************************************************************************
# For binary problem
class BinaryNet2(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(BinaryNet2, self).__init__()
    self.linear1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(hidden_size, 1)

  def forward(self, x):
    out = self.linear1(x)
    out = self.relu(out)
    out = self.linear2(out)
    # sigmoid at the end
    y_pred = torch.sigmoid(out)
    return y_pred

  # OR
  def forward(self, X):
    out = torch.relu(self.linear1(x))
    out = torch.sigmoid(self.linear2(out))
    return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.BCELoss() # (binary classification does not require softmax)


# vi. ACTIVATION FUNCTIONS
#**********************************************************************************
nn.Sigmoid
nn.Softmax
nn.Tanh
nn.LeakyReLU

# available in PyTorch API as
torch.softmax
torch.tanh

# sometimes they are available in torch.nn.functional 
import torch.nn.functional as F

F.leaky_relu # this is only available intorch.nn.functional 
F.relu # same as torch.relu


# v. FEED FORWARD NEURAL NETWORK
#**********************************************************************************

