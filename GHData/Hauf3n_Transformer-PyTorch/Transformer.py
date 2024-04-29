import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Encoder import Encoder
from Decoder import Decoder

device = torch.device("cuda:0")
dtype = torch.float

# embedding size of fasttext models
d_model = 300 

class Transformer(nn.Module):
    
    def __init__(self, cells, heads, seq_len_enc, seq_len_dec, attention_dimensions, vocab_size):
        super().__init__()
        
        self.encoder = Encoder(cells, heads, seq_len_enc, attention_dimensions).to(device)
        self.decoder = Decoder(cells, heads, seq_len_dec, attention_dimensions, vocab_size).to(device)
    
    def forward(self, x_encoder, x_decoder):
        # x_in shape: batch_size, seq_len_in, d_model
        # x_out shape: batch_size, seq_len_out, d_model
        
        encoder_k, encoder_v = self.encoder(x_encoder)
        out = self.decoder(x_decoder, encoder_k, encoder_v)
        
        # output needs softmax afterwards | cross_entropy_loss or F.softmax
        return out