import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def event_baseline(X, n_particles_per_event=6, input_dim=4):
    ## X contains all events where each event has many jets, input_dim=4 (if taking 4 vectors/ 4 features)
    features=[]
    for e in X:
        features.append(e[:n_particles_per_event])
        
    h_jets = torch.cat(list(torch.cuda.FloatTensor(np.asarray(features))))
    h_jets = h_jets.reshape(len(X), n_particles_per_event, -1)

    return h_jets.view(-1,n_particles_per_event*input_dim)


class ThreeLayerNet(nn.Module):
    def __init__(self, input_dim, hidden, n_particles_per_event ):
   
       super(ThreeLayerNet, self).__init__()
       activation_string = 'relu'
       self.activation = getattr(F, activation_string)
       
       ### 3  fully connected layers fc1, fc2, fc3
       self.fc1 = nn.Linear(n_particles_per_event*input_dim, hidden)
       self.fc2 = nn.Linear(hidden, hidden)
       self.fc3 = nn.Linear(hidden, 1)
    
       #### weights initialization#######
       gain = nn.init.calculate_gain(activation_string)
       nn.init.xavier_uniform(self.fc1.weight, gain=gain)
       nn.init.xavier_uniform(self.fc2.weight, gain=gain)
       nn.init.xavier_uniform(self.fc3.weight, gain=gain)
       nn.init.constant(self.fc3.bias, 1)
       
    def forward(self, X):
        out_stuff=event_baseline(X)
        h = self.fc1(out_stuff)
        h = self.activation(h)
        h = self.fc2(h)
        h = self.activation(h)
        h = self.fc3(h)
        h = F.sigmoid(self.fc3(h))
        return h
      

def log_loss(y, y_pred):
    return F.binary_cross_entropy(y_pred, y)

