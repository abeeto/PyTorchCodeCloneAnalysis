from Decoder import *
from Encoder import *

import torch 
import numpy as np 
import torch.nn.functional as F 
import torch.nn as nn 


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, state):
        q_z = self.encoder(state)
        z = q_z.rsample()
        return self.decoder(z), q_z
