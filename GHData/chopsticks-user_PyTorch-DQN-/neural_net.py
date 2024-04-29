import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from collections import deque
import os

class net(nn.Module):
    def __init__(self, in_features = 1, out_features = 1, hidden_layers_dims = [1], learning_rate = 0.001,
                    file_path = None):
        super().__init__()
        #self.device = "cpu"
        self.device = T.device("cuda")
        #learning rate??
        #use named_tuple -> done
        #exception when hidden_layers_sizes contains any element with a value of 0
        #hidden layers are hidden from named_parameters call -> done
        #be mindful of NaN values
        #still using cpu -> done
        #might need to move first_layer and final_layer in to a list contains all layers -> done
        #might not need in/out_features_dim (but out_features needs to be 
        #converted back to its orginial dimension)
        #self.to(self.device) might not be needed
        #edit Adam optim -> done

        if file_path:
            self.load(file_path)
            return

        self.lr = learning_rate
        in_features = int(np.prod(in_features))

        hidden_layers_dims[:0] = [in_features]
        hidden_layers_dims[len(hidden_layers_dims):] = [out_features]
        self.module = nn.ModuleList([nn.Linear(hidden_layers_dims[i], hidden_layers_dims[i + 1]).to(device = self.device)
                        for i in range(len(hidden_layers_dims) - 1)])

        self.loss = nn.MSELoss()
        self.optimizer = O.Adam(self.parameters(), lr = self.lr)
        self.to(self.device)
        return

    #exception when hidden_layers_sizes contains any element with a value of 0
    #if passed sample has input dimenson != in_features, return dimension error 
    #(may caused by not initilzing in_features)
    def forward(self, x):
        for i in range(len(self.module) - 1):
            x = F.relu(self.module[i](x.to(self.device)))
        x = self.module[-1](x)
        return x

    def attributes(self):
        for p in self.named_parameters():
            print(p)

    def save(self, path):
        a = 1

    def load(self, path):
        a = 1







