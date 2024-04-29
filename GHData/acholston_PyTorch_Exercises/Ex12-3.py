# Lab Ex12-3
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

#########################################################################################
#											#
# Word embedding allows better use of the function space by converting to a vector	#
# 											#
# Rather than using a single one-hot value for hidden state interpretation, it allows a #
# learnable functon to use the whole vector - Representing information more efficiently #				
#											#
#########################################################################################


torch.manual_seed(777)  # reproducibility


idx2char = ['h', 'i', 'e', 'l', 'o']

#remove 1-hot coding
x = [[0, 1, 0, 2, 3, 3]]

y = [1, 0, 2, 3, 3, 4]

num_classes = 5
input_size = 5  #Number of possible inputs for embedding
hidden_size = 5
embed_size = 5
batch_size = 1 
sequence_length = 6 
num_layers = 1 


class RNN(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers, embed_size):
        super(RNN, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

	#Create Embedding
        self.emb = nn.Embedding(input_size, embed_size)

	#Create RNN cell
        self.rnn = nn.RNN(input_size=embed_size, hidden_size=hidden_size, batch_first=True)

	#Final linear layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
	#get embeddings
        x = self.emb(x)
        
        # Initialize hidden state
        h_0 = Variable(torch.zeros(
            x.size(0), self.num_layers, self.hidden_size))

        # Reshape input
        x = x.view(x.size(0), self.sequence_length, self.input_size)

	# Use RNN block
        x, h = self.rnn(x, h_0)

	# Reshape and use linear 
        x = x.view(-1, self.num_classes)
        x = self.fc(x)
        return F.log_softmax(x)
		

# Instantiate RNN model
model = RNN(num_classes, input_size, hidden_size, num_layers, embed_size)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    inputs = Variable(torch.LongTensor(x))
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
