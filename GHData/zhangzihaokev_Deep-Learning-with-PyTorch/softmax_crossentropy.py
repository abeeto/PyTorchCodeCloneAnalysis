from audioop import cross
import torch
import torch.nn as nn
import numpy as np

# manually defined softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# manually defined cross entropy loss
def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
# print(outputs)

# nn.CrossEntrpyLoss applied nn.LogSoftmax and nn.NLLLoss (negative log likelihood loss) --> do not apply softmax in last layer
# y has class labels, not One-Hot, only put the correct class label, e.g. before: [1,0,0] now: [0]
# y_pred has raw scores, no Softmax
loss = nn.CrossEntropyLoss()

# 3 samples
Y = torch.tensor([2, 0, 1]) 
# size n_sample x n_classes, so we have 3x3 tensors 
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [1.0, 3.0, 0.1]]) # raw values without softmax
Y_pred_bad = torch.tensor([[4.5, 2.0, 0.3], [0.5, 2.0, 3.3], [4.5, 2.0, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item(), l2.item())

_, pred1 = torch.max(Y_pred_good, 1)
_, pred2 = torch.max(Y_pred_bad, 1)

print(pred1, pred2)

# multiclass classifcation
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes) -> None:
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() # activation function
        self.linear2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax at the end
        return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()

# binary classification
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() # activation function
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # apply sigmoid 
        y_pred = torch.sigmoid(out)
        return y_pred

model = NeuralNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELOss()