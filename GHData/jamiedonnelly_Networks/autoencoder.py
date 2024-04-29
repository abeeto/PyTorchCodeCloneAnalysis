
import torch 
from torch import nn 
from torch.functional import F

class AutoEnc(nn.Module):
  
    """
      Template for simple autoencoder. Can alter for purpose. Class contains both a encode and decode method. 
      The __call__ method for forward passes has been rewritten to explicity outline how the network functions. 
    """

    def __init__(self,input_dim,latent_dim):
        super(AutoEnc,self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.enc_fc1 = nn.Linear(self.input_dim,256)
        self.enc_fc2 = nn.Linear(256,self.latent_dim)
        self.dec_fc1 = nn.Linear(self.latent_dim,256)
        self.dec_fc2 = nn.Linear(256,self.input_dim)
        
    def encode(self,x):
        x = self.enc_fc1(x)
        x = F.relu(x)
        x = self.enc_fc2(x)
        return x    

    def decode(self,x):
        x = self.dec_fc1(x)
        x = F.relu(x)
        x = self.dec_fc2(x)
        return x
    
    def __call__(self,x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded 
