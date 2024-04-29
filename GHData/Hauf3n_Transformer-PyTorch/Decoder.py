import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Multi_Head_Attention import Multi_Head_Attention
from Encoder_Decoder_Attention import Encoder_Decoder_Attention

device = torch.device("cuda:0")
dtype = torch.float

# embedding size of fasttext models
d_model = 300 

class Decoder_Cell(nn.Module):

    def __init__(self, heads, seq_len, attention_dimension, ff_inner=2048):
        super().__init__()
        self.seq_len = seq_len
        self.heads = heads
        
        # construct decoder mask
        a = np.triu(np.ones((seq_len,seq_len)), k=1) 
        self.mask = torch.unsqueeze(torch.tensor(a).to(device).long(),dim=0)
        
        self.self_attention = Multi_Head_Attention(heads, seq_len, attention_dimension).to(device)
        self.enc_dec_attention = Encoder_Decoder_Attention(heads, seq_len, attention_dimension).to(device)
        
        self.layer_norm_1 = nn.LayerNorm([d_model])
        self.layer_norm_2 = nn.LayerNorm([d_model])
        self.layer_norm_3 = nn.LayerNorm([d_model])
        
        ff_network = [
            nn.Linear(d_model, ff_inner),
            nn.ReLU(),
            nn.Linear(ff_inner, d_model),
        ]
        self.feed_forward_net = nn.Sequential(*ff_network)
        
    def forward(self, x, encoder_k, encoder_v):
        # x shape: batch,seq_len,d_model
        batch_size, _, _ = x.shape
        
        # self attention
        mask = self.mask.repeat(batch_size*self.heads,1,1)
        z_1 = self.self_attention(x, mask)
        
        # 1st residual
        residual_1 = x + z_1
        # 1st norm
        norm_1 = self.layer_norm_1(residual_1)
        
        # encoder-decoder attention
        z_2 = self.enc_dec_attention(norm_1, encoder_k, encoder_v)
        
        # 2nd residual
        residual_2 = norm_1 + z_2
        # 2nd norm
        norm_2 = self.layer_norm_2(residual_2)
        
        # reshape norm for feed forward network
        ff_in = torch.reshape(norm_2, (batch_size*self.seq_len, d_model))
        # feed forward
        ff_out = self.feed_forward_net(ff_in)
        # reshape back
        ff_out = torch.reshape(ff_out, (batch_size, self.seq_len, d_model))
        
        # 3rd residual
        residual_3 = norm_2 + ff_out
        # 3rd norm
        norm_3 = self.layer_norm_3(residual_3)
        
        return norm_3
        
class Decoder(nn.Module):
    
    def __init__(self, cells, heads, seq_len, attention_dimensions, vocab_size):
        super().__init__()
        self.heads = heads
        self.seq_len = seq_len
        self.attention_dimensions = attention_dimensions
        self.vocab_size = vocab_size
        
        # stacked encoder cells
        self.decoder_cells = [ Decoder_Cell(heads, seq_len, attention_dimensions).to(device) for i in range(cells)]
        
        # output layer and then softmax
        self.final_linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, encoder_k, encoder_v):
        
        batch_size = x.shape[0]
        
        for decoder_cell in self.decoder_cells:
            x = decoder_cell(x,encoder_k, encoder_v)
            
        # reshape for linear
        x = torch.reshape(x, (batch_size*self.seq_len, d_model))
        
        # feed in final layer
        x = self.final_linear(x)
          
        return x