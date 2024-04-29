# Lab 12 RNN
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

#################################################################
# 								#
# Splits the sentence into smaller segments 			#
#								#
# Can be useful if horizon on the important previous states is  #
# small 		(can be less stable)			#
#								#
#################################################################


torch.manual_seed(777)  # reproducibility


sent = "if you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea."

y = []
x = [[]]
check = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', ',', '\'', '.']

for i in sent:
    if ord(i) >= 97:
        y.append(ord(i)-97)
    else:
        if i == ' ':
            y.append(26)
        elif i == ',':
            y.append(27)
        elif i == '\'':
            y.append(28)
        elif i == '.':
            y.append(29)
    if len(x[0]) > 0:
        x[0].append(np.random.randint(0, 20))
    else:
	#Generate some kind of starting token
	x[0].append(25)

num_classes = len(check)
input_size = 30  
embed_size = 300
hidden_size = 300 
batch_size = 1 
sequence_length = 6  # smaller than overall sequence length
num_layers = 1  # one-layer rnn


class RNN(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, embed_size):
        super(RNN, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
	self.embed_size = embed_size
        self.sequence_length = sequence_length

	#Setup embedding
        self.emb = nn.Embedding(num_classes, embed_size)

	#RNN Cell
        self.rnn = nn.RNN(input_size=embed_size, hidden_size=hidden_size, batch_first=True)

	#FC layers
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
	#get embeddings
        x = self.emb(x)
        
        # Initialize hidden states
        h_0 = Variable(torch.zeros(
            1, self.num_layers, self.hidden_size))

        # Reshape input
        x = x.view(1, self.sequence_length, self.embed_size)

	# Run through RNN
        x, h = self.rnn(x, h_0)

	# Two fully connected Layers
        x = x.view(-1, hidden_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)
		

# Instantiate RNN model
model = RNN(num_classes, input_size, hidden_size, num_layers, embed_size)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    #Overall test sentece
    inputs0 = Variable(torch.LongTensor(x)) 
    for i in range(100):
	#Split into random smaller segments
        num = np.random.randint(0, len(sent)-sequence_length+1)
        x0 = x[0][num:num+sequence_length]
        y0 = y[num:num+sequence_length]
        inputs = Variable(torch.LongTensor(x0))
        labels = Variable(torch.LongTensor(y0))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            model.sequence_length = len(sent)
            outputs = model(inputs0)
            _, idx = outputs.data.max(1)
            result_str = [check[c] for c in idx.squeeze()]
            print("Predicted string: ", ''.join(result_str))
            model.sequence_length = sequence_length

for epoch in range(1, 1000):
    train(epoch)
