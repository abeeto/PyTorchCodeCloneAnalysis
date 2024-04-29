# Lab Exercise 13-1 - Implement NMT
# Model structure follows from Neural Machine Translation by Jointly Learning to Align and Translate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


#Short Spanish Test Case
x_data = [[0, 6, 2], [0, 4, 2], [0, 1, 3], [0, 4, 3], [0, 1, 5]]
y_data = [[0, 2, 6], [0, 2, 4], [0, 1, 5], [0, 1, 4], [0, 3, 5]]

idx = ["el", "perro", "azul", "rojo", "gato", "amarillo", "cabello"]
idy = ["the", "red", "blue", "yellow", "cat", "dog", "horse"]

x_test = [0, 1, 2]
#Test => 0, 1, 2 (The blue dog)

num_cases = 5
input_size = 7 
hidden_size = 100 
embed_size = 100
batch_size = 1 
sequence_length = 3  
num_layers = 1  


#For generating initial sentence encoding
class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, embed_size):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

	#Feed through RNN
	
	#Embedding
        self.emb = nn.Embedding(self.input_size, embed_size)

	#RNN Module - Bidirectional for runn sentence both ways
	self.gru = nn.GRU(embed_size, self.hidden_size, bidirectional=True)

	#Initialize the hidden state of the decoder here
	self.init_s = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
	#Embed and processes all sequences
        x = self.emb(x)
	x, h = self.gru(x, h)

	#Generate input hidden state for decoder (from h1<-)
	s = self.init_s(h[0][0]).view(-1, self.hidden_size)

	#Concatenate hidden state for future use
	h = h.transpose(0, 1).contiguous()
	h = h.view(-1, self.hidden_size*2)

        return x, h, s


#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, embed_size):
	super(DecoderRNN, self).__init__()
	self.hidden_size = hidden_size
	self.sequence_length = sequence_length
	self.input_size = input_size

	#Weights for conversion of hidden states to energies
	self.U = nn.Linear(hidden_size*2, hidden_size)
	self.W = nn.Linear(hidden_size, hidden_size)

	self.v = nn.Linear(hidden_size, 1)

	#Conversion of inputs to RNN
	self.Uc = nn.Linear(hidden_size*2, hidden_size)
	self.Wy = nn.Linear(hidden_size, hidden_size)

	#Embedding/GRU
	self.emb = nn.Embedding(self.input_size, embed_size)
	self.gru = nn.GRU(self.hidden_size, hidden_size)

	#Final class reduction
	self.fc = nn.Linear(self.hidden_size, self.input_size)

    #Performing attention
    def attn(self, s, h):
	#Initiallize context vector to size of 2n
	c = Variable(torch.zeros(self.hidden_size*2))

	energy = []	
	#For each of the sequence
	for i in range(self.sequence_length):
	    #Energy values
	    temp = F.tanh(self.U(h[i]) + self.W(s))
	    energy.append(temp)
	energy = torch.stack(energy)
	energy = energy.view(self.hidden_size, -1)
	
	#Get alpha
	alpha = energy.transpose(0, 1).contiguous()
	alpha = self.v(alpha)
	alpha = F.softmax(alpha)

	for i in range(self.sequence_length):
	    #multiply - output 1x2*n (add for each sequence value)
	    temp = torch.mul(alpha[i].view(-1), h[i].view(-1))
	    c += temp.view(self.hidden_size*2)

	return c

    def forward(self, y, h, s):
	#Embedding
	y = self.emb(y)

	#Context Vector from attention
	c = self.attn(s, h).view(-1, self.hidden_size*2)
	
	#Generate input to rnn (embedding and context)
	y = self.Wy(y) + self.Uc(c)
	y = y.view(1, 1, -1)
	s = s.view(1, 1, -1)

	#RNN
	y, s = self.gru(y, s)

	#Generate output words
	y = y.squeeze(0)
	y = self.fc(y)

	y = F.log_softmax(y)

	return y, s
	


# Instantiate RNN model
enc = EncoderRNN(input_size, hidden_size, embed_size)
dec = DecoderRNN(input_size, hidden_size, sequence_length, embed_size)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=0.001)


#Training
def train():
    for i in range(len(x_data)):
        inputs = Variable(torch.LongTensor([x_data[i]]))
        labels = Variable(torch.LongTensor(y_data[i]))

	#Generate initial hidden state
        h = Variable(torch.zeros(2, sequence_length, hidden_size))

        outputs = []

	#Run encoder over whole sequence
        out, h, s = enc(inputs, h)

	#Perform over each sequence value
        for i in range(sequence_length):
	    #If first sequence set output as SOS
	    if i > 0:
	        _, idx = output.max(1)
	        inp = Variable(torch.LongTensor(idx.data))
	    else:
	        inp = Variable(torch.LongTensor([0]))
	    output, s = dec(inp, h, s)
            outputs.append(output[0])

        outputs = torch.stack(outputs)

	#Train
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

#Test
def test():
    inputs = Variable(torch.LongTensor([x_test]))
    #Generate first hidden_state
    h = Variable(torch.zeros(2, sequence_length, hidden_size))
    outputs = []

    #Run encoder over whole sequence
    out, h, s = enc(inputs, h)

    #perform over each value in sequence
    for i in range(sequence_length):
	if i > 0:
	    _, idx = output.max(1)
	    inp = Variable(torch.LongTensor(idx.data))
	else:
	    inp = Variable(torch.LongTensor([0]))
	output, s = dec(inp, h, s)
        outputs.append(output[0])

    outputs = torch.stack(outputs)

    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    result_str = [idy[c] for c in idx.squeeze()]
    print("Predicted string: ", ' '.join(result_str))


for i in range(50):
    train()
    test()






















