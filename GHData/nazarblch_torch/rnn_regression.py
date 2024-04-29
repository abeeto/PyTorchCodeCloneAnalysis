
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from torch.nn import Sequential
from torch.nn import Linear
from torch.nn import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from hart_gen import gen_hart

numpy.random.seed(7)


# dataframe = pandas.read_csv('data/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
# dataset = dataframe.values
# dataset = dataset.astype('float32')

dataset = gen_hart(30).reshape(-1,1)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)



# split into train and test sets
train_size = int(len(dataset) * 0.99)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t and Y=t+1
look_back = 100
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

device = torch.device("cuda")
trainX = torch.tensor(trainX).to(device)
trainY = torch.tensor(trainY).to(device)


class LSTMReg(nn.Module):

    def __init__(self, embedding_dim, hidden_dim,  tagset_size):
        super(LSTMReg, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 1)
        self.tagset_size = tagset_size

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim, device=device),
                torch.zeros(1, 1, self.hidden_dim, device=device))

    def forward(self, sentence):

        lstm_out, self.hidden = self.lstm(sentence.view(len(sentence), 1, -1), self.hidden)

        y = self.hidden2tag(lstm_out[len(sentence)-1].view(self.hidden_dim))

        res = y
        for i in range(10):
            lstm_out, self.hidden = self.lstm(y.view(1, 1, -1), self.hidden)
            y = self.hidden2tag(lstm_out.view(self.hidden_dim))
            res = torch.cat((res, y))

        return res


model = LSTMReg(1, 100, 1).to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
    for t in range(train_size-200):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()


        # Step 3. Run our forward pass.
        x = trainX[t]
        tag_scores = model(x)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores.reshape(11), trainY[t: t+11])
        loss.backward()
        optimizer.step()
        print(loss)


preds = trainX[0].cpu().numpy()

with torch.no_grad():
    model.init_hidden()

    for t in range(len(trainX)-1):
      x = torch.tensor(preds[len(preds) - len(trainX[0]):], device=device)
      y = model(x).reshape(11).cpu().numpy()
      preds = numpy.append(preds, y)


plt.plot(preds)
plt.show()