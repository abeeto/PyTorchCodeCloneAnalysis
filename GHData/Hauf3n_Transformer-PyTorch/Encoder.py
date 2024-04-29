import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Multi_Head_Attention import Multi_Head_Attention

device = torch.device("cuda:0")
dtype = torch.float

# embedding size of fasttext models
d_model = 300 

class Encoder_Cell(nn.Module):
    
    def __init__(self, heads, seq_len, attention_dimension, ff_inner=2048):
        super().__init__()
        self.seq_len = seq_len
        
        self.self_attention = Multi_Head_Attention(heads, seq_len, attention_dimension)
        self.layer_norm_1 = nn.LayerNorm([d_model])
        self.layer_norm_2 = nn.LayerNorm([d_model])
        
        ff_network = [
            nn.Linear(d_model, ff_inner),
            nn.ReLU(),
            nn.Linear(ff_inner, d_model),
        ]
        self.feed_forward_net = nn.Sequential(*ff_network)
        
    def forward(self, x):
        # x shape: batch,seq_len,d_model
        batch_size, _, _ = x.shape
        
        # self attention
        z = self.self_attention(x)
        
        # 1st residual
        residual_1 = x + z
        # 1st norm
        norm_1 = self.layer_norm_1(residual_1)
        
        # reshape norm for feed forward network
        ff_in = torch.reshape(norm_1, (batch_size*self.seq_len, d_model))
        # feed forward
        ff_out = self.feed_forward_net(ff_in)
        # reshape back
        ff_out = torch.reshape(ff_out, (batch_size, self.seq_len, d_model))
        
        # 2nd residual
        residual_2 = norm_1 + ff_out
        # 2nd norm
        norm_2 = self.layer_norm_1(residual_2)
        
        return norm_2
        
class Encoder(nn.Module):
    
    def __init__(self, cells, heads, seq_len, attention_dimensions):
        super().__init__()
        self.heads = heads
        self.seq_len = seq_len
        self.attention_dimensions = attention_dimensions
        
        # stacked encoder cells
        encoder_cells = [ Encoder_Cell(heads, seq_len, attention_dimensions).to(device) for i in range(cells)]
        self.encode = nn.Sequential(*encoder_cells)
        
        # key and value output of encoder
        self.kv = nn.Linear(d_model, attention_dimensions * heads * 2)
    
    def forward(self, x):
        
        batch_size = x.shape[0]
        
        # encoding shape: batch_size, seq_len, d_model
        encoding = self.encode(x) 
        
        # reshape to feed into linear kv layer
        encoding = torch.reshape(encoding, (batch_size * self.seq_len, d_model))
        
        # apply linear
        kv = self.kv(encoding)
        # reshape back
        kv = torch.reshape(kv, (batch_size, self.seq_len, self.attention_dimensions * self.heads * 2))
        
        # seperate k and v
        kv = torch.reshape(kv, (batch_size, self.seq_len, self.heads, 2, self.attention_dimensions))
        
        # permute head to front for parallel processing
        kv = kv.permute(0,2,1,3,4)
        
        # split k, v
        k = kv[:,:,:,0,:]
        v = kv[:,:,:,1,:]
        
        # fuse batch_size and head dim for parallel processing
        k = torch.reshape(k, (batch_size * self.heads, self.seq_len, self.attention_dimensions))
        v = torch.reshape(v, (batch_size * self.heads, self.seq_len, self.attention_dimensions))
        
        return k, v