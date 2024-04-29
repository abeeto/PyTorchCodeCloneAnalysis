# Lab 12-5 generate RNN manually
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(777)  # reproducibility


idx2char = ['h', 'i', 'e', 'l', 'o']

#Create one-hot codings
x_data = [[0, 1, 0, 2, 3, 3]]   # hihell
x_one_hot = [[[1, 0, 0, 0, 0],   # h 0
              [0, 1, 0, 0, 0],   # i 1
              [1, 0, 0, 0, 0],   # h 0
              [0, 0, 1, 0, 0],   # e 2
              [0, 0, 0, 1, 0],   # l 3
              [0, 0, 0, 1, 0]]]  # l 3

y_data = [1, 0, 2, 3, 3, 4]    # ihello

labels = Variable(torch.LongTensor(y_data))

num_classes = 5
input_size = 5  # one-hot size
hidden_size = 10  
batch_size = 1   
sequence_length = 6  
num_layers = 1  


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

	#Input connection to hidden
        self.fci = nn.Linear(input_size, hidden_size)

	#Hidden to hidden
	self.fch = nn.Linear(hidden_size, hidden_size)
	
	#Hidden to output
	self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, h):
	#Perform hidden layer
        h = self.fci(x) + F.relu(self.fch(h))

	#Output layer
	x = F.relu(self.fc(h))
        x = x.view(-1, num_classes)
	x = F.log_softmax(x)

        return x, h


# Instantiate RNN model
rnn = RNN(input_size, hidden_size)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.1)

# Train the model
for epoch in range(100):
    hidden = Variable(torch.zeros(1, 1, hidden_size))
    outputs = []

    #Iterate through inputs to generate each hidden state and output
    for i in range(sequence_length):
        inputs = Variable(torch.Tensor([[x_one_hot[0][i]]])) 
        out, hidden = rnn(inputs, hidden)
        outputs.append(out[0])
    outputs = torch.stack(outputs, dim=0)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idx2char[c] for c in idx.squeeze()]
    if epoch % 10 == 0:
        print("epoch: %d, loss: %1.3f" % (epoch, loss.data[0]))
        print("Predicted string: ", ''.join(result_str))

