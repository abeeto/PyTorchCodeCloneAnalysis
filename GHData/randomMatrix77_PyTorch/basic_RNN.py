import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

# one hot encoding

idx2char = ['h','e','l','o']

x_data = [0,1,2,2,3]

one_hot = np.eye(4)[x_data]

y_data = [0,1,2,2,3]

inputs = Variable(torch.Tensor(one_hot))
labels = Variable(torch.LongTensor(y_data))

# Hyper Parameters

input_size = 4
num_classes = 4
hidden_size = 4
batch_size = 1
seq_length = 1
num_layers = 1
lr = 0.1
epochs = 10

# Model

class RNN(nn.Module):

    def __init__(self, num_classes, batch_size, input_size, hidden_size, num_layers, seq_length):

        super(RNN, self).__init__()

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cell = nn.RNN(input_size = self.input_size, hidden_size = self.hidden_size, batch_first = True)

    def forward(self, x, hidden):

        x = x.view(self.batch_size, self.seq_length, self.input_size)

        out, hidden = self.cell(x, hidden)

        return out, hidden

model = RNN(num_classes, batch_size, input_size, hidden_size, num_layers, seq_length)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):

    loss = 0

    hidden = Variable(torch.zeros(1, batch_size, num_classes))

    for inp, lab in zip(inputs, labels):

        out, hidden = model(inp, hidden)

        out = out.view(1,4)
        lab = lab.view(1)

        _, idx = out.max(1)

        print('prediction: {0},  Actual:  {1}'.format(idx2char[idx], idx2char[lab.item()]))

        optimizer.zero_grad()

        loss += criterion(out, lab)

    loss.backward(retain_graph = True)

    optimizer.step()

    print(loss)



    










    
