import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class HighwayLayer(nn.Module):

    def __init__(self, input_size, bias=-1):
        super(HighwayLayer, self).__init__()
        self.plain_layer = nn.Linear(input_size, input_size)
        self.transform_layer = nn.Linear(input_size, input_size)
        self.transform_layer.bias.data.fill_(bias)
        
    def forward(self, x): 
        plain_layer_output = nn.functional.relu(self.plain_layer(x))  
        transform_layer_output = nn.functional.softmax(self.transform_layer(x))
        transform_value = torch.mul(plain_layer_output, transform_layer_output)
        carry_value = torch.mul((1 - transform_layer_output), x)
        return torch.add(carry_value, transform_value) 

class RecurrentHighwayNetwork(nn.Module):
    
    def __init__(self, in_features, rec_depth, num_layers):
        super(RecurrentHighwayNetwork, self).__init__()
         
        self.hidden_states = []
        self.rec_depth = rec_depth

        self.highway_layers = nn.ModuleList([ HighwayLayer(in_features) for _ in range(num_layers)])    
        
    def forward(self, sequence, hidden): 
        i = 0
        cur_depth = 0
        for x in sequence:
            for _ in range(0, self.rec_depth):
                if i==0 and cur_depth==0:
                    hidden[i] = x # Initially it is the input
                else:
                    hidden[i] = hidden[i-1].clone() # Otherwise the previous time-step's output
            
                for layer in self.highway_layers: # This the recurrence over layers
                    hidden[i] = layer(hidden[i].clone())
                
                cur_depth = cur_depth + 1

            # Previous timestep's final output (i is the time step)
            i = i +1
                
                 
        return hidden, hidden

class LanguageModel(nn.Module):

    def __init__(self, ntoken, ninp, nhid, rec_depth, nlayers, dropout=0.5):
        super(LanguageModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = RecurrentHighwayNetwork(ninp, rec_depth, nlayers)
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()

        self.decoder.weight = self.encoder.weight

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))  
        #emb = self.encoder(input) # if Not using dropout at the input
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output) 
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden


