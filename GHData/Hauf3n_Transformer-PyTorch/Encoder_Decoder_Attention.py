import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0")
dtype = torch.float

# embedding size of fasttext models
d_model = 300 

class Encoder_Decoder_Attention(nn.Module):

    def __init__(self, heads, seq_len, attention_dimensions):
        super().__init__()
        self.heads = heads
        self.seq_len = seq_len
        self.attention_dimensions = attention_dimensions
        
        self.q = nn.Linear(d_model, attention_dimensions * heads).to(device)
        self.final_linear = nn.Linear(heads * attention_dimensions, d_model).to(device)
        
    def forward(self, x, encoder_k, encoder_v):
        # x shape: batch_size, seq_len, d_model
        # encoder k/v shape: batch_size*heads, seq_len, attention_dimensions
        batch_size = x.shape[0]
        
        # reshape for linear q layer
        x = torch.reshape(x, (batch_size*self.seq_len, d_model))
        
        # compute q for every head
        q = self.q(x) # (seq * batch, heads * attention_dimensions)

        # reshape into (batch_size,seq_len,...)
        q = torch.reshape(q, (batch_size, self.seq_len, self.heads * self.attention_dimensions))
        # split into heads 
        q = torch.reshape(q, (batch_size, self.seq_len, self.heads, self.attention_dimensions))
        
        # permute head to front for parallel processing
        q = q.permute(0,2,1,3)
        
        # fuse batch_size and head dim for parallel processing
        q = torch.reshape(q, (batch_size * self.heads, self.seq_len, self.attention_dimensions))
        
        # transpose k
        k = torch.transpose(encoder_k, 1, 2)
        
        # multiply q and k
        qk = torch.bmm(q,k)
        # scale
        qk = qk / torch.sqrt(torch.tensor(self.attention_dimensions).to(device).to(dtype))
        # softmax
        qk = F.softmax(qk, dim=2)     
        
        # multiply with v
        qkv = torch.bmm(qk, encoder_v)
        
        # reshape to cat heads
        qkv = torch.reshape(qkv, (batch_size, self.heads, self.seq_len, self.attention_dimensions))
        # cat all heads
        qkv = qkv.permute(0,2,1,3)
        qkv = torch.reshape(qkv, (batch_size, self.seq_len, self.heads * self.attention_dimensions))
        
        # reshape to multiply with final linear
        qkv = torch.reshape(qkv, (batch_size * self.seq_len, self.heads * self.attention_dimensions))
        # multiply with final linear
        z = self.final_linear(qkv)
        
        # reshape to input format
        z = torch.reshape(z, (batch_size, self.seq_len, d_model))
        
        return z