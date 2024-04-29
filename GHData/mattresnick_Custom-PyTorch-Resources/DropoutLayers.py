'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

# Implementation of the common-practice standard dropout.
class InvertedDropout(nn.Module):
    def __init__(self, p=0.5, train=False):
        super(InvertedDropout, self).__init__()
        self.p = p

    def forward(self, X):
        if self.training:
            rv_dist = torch.distributions.bernoulli.Bernoulli(probs=1-self.p)
            return X*rv_dist.sample(X.size()).to(device)*(1.0/(1-self.p))
        return X

# Implementation of standard dropout as originally described.
class StandardDropout(nn.Module):
    def __init__(self, p=0.5, train=False):
        super(StandardDropout, self).__init__()
        self.q = 1-p

    def forward(self, X):
        if self.training:
            rv_dist = torch.distributions.bernoulli.Bernoulli(probs=self.q)
            return X*rv_dist.sample(X.size()).to(device)
        return X*self.q

# Shift-dropout, as described in SERLU paper.
class ShiftDropout(nn.Module):
    def __init__(self, p=0.5, train=False):
        super(ShiftDropout, self).__init__()
        self.p = p
        self.f_min = (-1)*(1.07862)*(2.90427)*np.exp(-1)

    def forward(self, X):
        if self.training:
            # Obtain a bernoulli sample for all neurons, with probability 1-p of staying on.
            rv_dist = torch.distributions.bernoulli.Bernoulli(probs=1-self.p)
            rv_sample = rv_dist.sample(X.size()).to(device)
            
            # Shutoff neurons where appropriate, temporarily.
            rv_mask = X*rv_sample
            
            # Create a mask where all zero positions are now f_min.
            inverse_mask = (rv_mask==0).type(torch.cuda.FloatTensor)*self.f_min
            
            # Replace all zeros in original tensor by adding mask.
            z_tilde = rv_mask + inverse_mask
            
            # Preserve mean of z_tilde via z_hat.
            z_hat = (z_tilde - self.p*self.f_min)*(1.0/(1-self.p))
            
            return z_hat
        
        return X



def createDropoutLayers(DO_type, num_layers, p=0.5):
    # Given a list of probabilities, make each layer accordingly.
    # Given a single value, make all layers based on that value.
    if type(p) is list:
        dropout_params = p
    else:
        dropout_params = np.repeat(p, num_layers)
    
    # Create layers based on given type and prob(s).
    if DO_type=='alpha':
        DO_layers = nn.ModuleList([nn.AlphaDropout(p=do_p) \
            for do_p in dropout_params])
    elif DO_type=='shift':
        DO_layers = nn.ModuleList([ShiftDropout(p=do_p) \
            for do_p in dropout_params])
    elif DO_type=='inverse':
        DO_layers = nn.ModuleList([InvertedDropout(p=do_p) \
            for do_p in dropout_params])
    elif DO_type=='custom_standard':
        DO_layers = nn.ModuleList([StandardDropout(p=do_p) \
            for do_p in dropout_params])
    elif DO_type=='standard':
        DO_layers = nn.ModuleList([nn.Dropout(p=do_p) \
            for do_p in dropout_params])
    else:
        DO_layers = nn.ModuleList([nn.Dropout(p=0) \
            for do_p in dropout_params])
    
    return DO_layers
