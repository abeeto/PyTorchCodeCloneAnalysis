# Lab 12-2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

#########################################################################################
#											#
# RNN is able to use sequence to predict next character - Rather than just current value#
# 											#
# Linear layers allowes separation between hidden state and output - Give more exact	#
# Interpretation for the output								#
#											#
#########################################################################################

torch.manual_seed(777)  # reproducibility


idx2char = ['h', 'i', 'e', 'l', 'o']

#Create input/output tests and one-hot coding
x = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[[1, 0, 0, 0, 0],
				[0, 1, 0, 0, 0],
				[1, 0, 0, 0, 0],
				[0, 0, 1, 0, 0],
				[0, 0, 0, 1, 0],
				[0, 0, 0, 1, 0]]]

y = [1, 0, 2, 3, 3, 4]

num_classes = 5
input_size = 5  
hidden_size = 5 
batch_size = 1  
sequence_length = 6 
num_layers = 1 


class RNN(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

	#Create RNN unit
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)

	#Combine with Linear Layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Init hidden state
        h_0 = Variable(torch.zeros(
            x.size(0), self.num_layers, self.hidden_size))

        # Reshape input
        x = x.view(x.size(0), self.sequence_length, self.input_size)

	# Use RNN
        x, h = self.rnn(x, h_0)
        x = x.view(-1, self.num_classes)
        x = self.fc(x)
        return F.log_softmax(x)
		

# Instantiate RNN model
model = RNN(num_classes, input_size, hidden_size, num_layers)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    inputs = Variable(torch.Tensor(x_one_hot))
    labels = Variable(torch.LongTensor(y))
    for i in range(100):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            _, idx = outputs.data.max(1)
            result_str = [idx2char[c] for c in idx.squeeze()]
            print("Predicted string: ", ''.join(result_str))

for epoch in range(1, 10):
    train(epoch)
