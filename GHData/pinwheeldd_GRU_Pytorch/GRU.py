import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import batch
import numpy as np
from utils import wrap,unwrap,wrap_X,unwrap_X

###########RecNN architecture for jet embedding#############
class RecNN(nn.Module):
    def __init__(self, n_features_embedding, n_hidden_embedding,**kwargs):
        super(RecNN,self).__init__()
        activation_string = 'relu'
        self.activation = getattr(F, activation_string)
        
        
        self.fc_u = nn.Linear(n_features_embedding, n_hidden_embedding)  ## W_u, b_u
        self.fc_h = nn.Linear(3 * n_hidden_embedding, n_hidden_embedding)## W_h, b_h
        
        gain = nn.init.calculate_gain(activation_string)
        nn.init.xavier_uniform(self.fc_u.weight, gain=gain)
        nn.init.orthogonal(self.fc_h.weight, gain=gain)


    def forward(self, jets):
        levels, children, n_inners, contents = batch(jets)
        n_levels = len(levels)
        embeddings = []
        
        for i, nodes in enumerate(levels[::-1]):
            j = n_levels - 1 - i
            inner = nodes[:n_inners[j]]
            outer = nodes[n_inners[j]:]
            u_k = self.fc_u(contents[j])
            u_k = self.activation(u_k) ##eq(3) in Louppe's paper
            
            if len(inner) > 0:
                zero = torch.zeros(1).long(); one = torch.ones(1).long()
                if torch.cuda.is_available(): zero = zero.cuda(); one = one.cuda()
                h_L = embeddings[-1][children[inner, zero]]
                h_R = embeddings[-1][children[inner, one]]
                
                h = torch.cat((h_L, h_R, u_k[:n_inners[j]]), 1)
                h = self.fc_h(h)
                h = self.activation(h)
                embeddings.append(torch.cat((h, u_k[n_inners[j]:]), 0))
            else:
                embeddings.append(u_k)
                    
        return embeddings[-1].view((len(jets), -1))

#######Building GRU architecture for full event##############
class GRU(nn.Module):
    def __init__(self, n_features_embedding,
                 n_hidden_embedding,
                 n_features_rnn,
                 n_hidden_rnn,
                 n_jets_per_event):
        
        super(GRU,self).__init__()
        activation_string = 'relu'
        self.activation = getattr(F, activation_string)
        RecNN_transform=RecNN
        self.transform = RecNN_transform(n_features_embedding, n_hidden_embedding)
        
        
        self.fc_zh=nn.Linear(n_hidden_rnn, n_hidden_rnn)  ## W_zh, b_zh
        self.fc_zx=nn.Linear(n_features_rnn, n_hidden_rnn) ## W_zx, b_zx
        self.fc_rh=nn.Linear(n_hidden_rnn, n_hidden_rnn)  ## W_rh, b_rh
        self.fc_rx=nn.Linear(n_features_rnn,n_hidden_rnn)  ## W_rx, b_rx
        self.fc_hh=nn.Linear(n_hidden_rnn, n_hidden_rnn)   ## W_hh, b_hh
        self.fc_hx=nn.Linear(n_features_rnn,n_hidden_rnn)  ## W_hx, b_hx
        
        gain = nn.init.calculate_gain(activation_string)
        nn.init.xavier_uniform(self.fc_zx.weight, gain=gain)
        nn.init.orthogonal(self.fc_zh.weight, gain=gain)
        nn.init.xavier_uniform(self.fc_rx.weight, gain=gain)
        nn.init.orthogonal(self.fc_rh.weight, gain=gain)
        nn.init.xavier_uniform(self.fc_hx.weight, gain=gain)
        nn.init.orthogonal(self.fc_hh.weight, gain=gain)
    
    def forward(self, X, n_jets_per_event=4):
        jets=[]
        features=[]
        
        for e in X:
            features.append(e[0][:n_jets_per_event])
            jets.extend(wrap_X(e[1][:n_jets_per_event]))   ## should be torch.tensor
        
        
        h_jets = torch.cat([torch.cat(list(torch.cuda.FloatTensor(np.asarray(features)))),self.transform(jets)],1)
        h_jets = h_jets.reshape(len(X), n_jets_per_event, -1)
        
        # GRU layer
        h = torch.zeros((len(X), self.fc_hh.bias.shape[0]))
        h = h.cuda()
        
        for t in range(n_jets_per_event):
            xt = h_jets[:, n_jets_per_event - 1 - t, :]
            zt = F.sigmoid(self.fc_zh(h)+self.fc_zx(xt))
            rt = F.sigmoid(self.fc_rh(h)+self.fc_rx(xt))
            ht = self.activation(self.fc_hh(torch.mul(rt, h))+self.fc_hx(xt))
            h = torch.mul(1. - zt, h) + torch.mul(zt, ht)

        return h


class Predict(nn.Module):
    def __init__(self, n_features_embedding,
                 n_hidden_embedding,
                 n_features_rnn,
                 n_hidden_rnn,
                 n_jets_per_event):
        super(Predict,self).__init__()
        self.transform_event = GRU(n_features_embedding,n_hidden_embedding,n_features_rnn,        n_hidden_rnn,n_jets_per_event)
        activation_string = 'relu'
        self.activation = getattr(F, activation_string)
       
        
        self.fc1 = nn.Linear(n_hidden_rnn, n_hidden_rnn)
        self.fc2 = nn.Linear(n_hidden_rnn, n_hidden_rnn)
        self.fc3 = nn.Linear(n_hidden_rnn, 1)
        
        gain = nn.init.calculate_gain(activation_string)
        nn.init.xavier_uniform(self.fc1.weight, gain=gain)
        nn.init.xavier_uniform(self.fc2.weight, gain=gain)
        nn.init.xavier_uniform(self.fc3.weight, gain=gain)
        nn.init.constant(self.fc3.bias, 1)
    
    
    def forward(self, X):
        out_stuff = self.transform_event(X)
        h = out_stuff
        h = self.fc1(h)
        h = self.activation(h)
        h = self.fc2(h)
        h = self.activation(h)
        h = F.sigmoid(self.fc3(h))
        return h


def square_error(y, y_pred):
    return (y - y_pred) ** 2

def log_loss(y, y_pred):
    return F.binary_cross_entropy(y_pred, y)


