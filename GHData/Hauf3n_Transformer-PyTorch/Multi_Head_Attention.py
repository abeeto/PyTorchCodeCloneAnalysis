import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0")
dtype = torch.float

# embedding size of fasttext models
d_model = 300 

class Multi_Head_Attention(nn.Module):
    
    def __init__(self, heads, seq_len, dimensions):
        super().__init__()
        
        self.heads = heads
        self.dimensions = dimensions
        self.seq_len = seq_len
        
        self.qkv = nn.Linear(d_model, self.dimensions * heads * 3)
        self.final_linear = nn.Linear(self.heads * self.dimensions, d_model)
    
    def forward(self, x, mask=None):
        # x shape: batch,seq_len,d_model
        batch_size, _, _ = x.shape
        
        # reshape for linear qkv layer
        x = torch.reshape(x, (batch_size*self.seq_len, d_model))
        
        # compute q,v,k for every head
        qkv = self.qkv(x)

        # reshape into (batch_size,seq_len,...)
        qkv = torch.reshape(qkv, (batch_size, self.seq_len, self.heads * 3 * self.dimensions))
        # split into heads and seperate q, k, v in different dims
        qkv = torch.reshape(qkv, (batch_size, self.seq_len, self.heads, 3, self.dimensions))
        
        # permute head to front for parallel processing
        qkv = qkv.permute(0,2,1,3,4)
        
        # extract q, k, v
        q = qkv[:,:,:,0,:]
        k = qkv[:,:,:,1,:]
        v = qkv[:,:,:,2,:]
        
        # fuse batch_size and head dim for parallel processing
        q = torch.reshape(q, (batch_size * self.heads, self.seq_len, self.dimensions))
        k = torch.reshape(k, (batch_size * self.heads, self.seq_len, self.dimensions))
        v = torch.reshape(v, (batch_size * self.heads, self.seq_len, self.dimensions))
        
        # transpose k
        k = torch.transpose(k, 1, 2)
        
        # multiply q and k
        qk = torch.bmm(q,k)
        # scale
        qk = qk / torch.sqrt(torch.tensor(self.dimensions).to(device).to(dtype))
        # optional masking
        if mask is not None:
            qk[mask == 1] = float('-inf')
        # softmax
        qk = F.softmax(qk, dim=2)
        
        # multiply with v
        qkv = torch.bmm(qk, v)
        
        # reshape to cat heads
        qkv = torch.reshape(qkv, (batch_size, self.heads, self.seq_len, self.dimensions))
        # cat all heads
        qkv = qkv.permute(0,2,1,3)
        qkv = torch.reshape(qkv, (batch_size, self.seq_len, self.heads * self.dimensions))
        
        # reshape to multiply with final linear
        qkv = torch.reshape(qkv, (batch_size * self.seq_len, self.heads * self.dimensions))
        # multiply with final linear
        z = self.final_linear(qkv)
        
        # reshape to input format
        z = torch.reshape(z, (batch_size, self.seq_len, d_model))
        
        return z