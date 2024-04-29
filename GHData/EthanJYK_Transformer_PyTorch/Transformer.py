#-----------------------------------------------------------------------------#
#                               Transformer                                   #
#                                                     July 2019, Junyoung Kim #
#                                                   Junyoung.JK.Kim@gmail.com #
#                                                                             #
# Reference:                                                                  #
# Attention is All You Need: https://arxiv.org/abs/1706.03762                 #
# Harvard NLP: http://nlp.seas.harvard.edu/2018/04/03/attention.html          #
# Samuel Lynn-Evans: https://towardsdatascience.com/                          #
#                    how-to-code-the-transformer-in-pytorch-24db27c8f9ec      #
#                                                                             #
#-----------------------------------------------------------------------------#


# Embedding ------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
import time
from torch.autograd import Variable


# Embedder
'''
vocab_size = vocabulary size
d_model = dimension of Embedding vector
'''
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx=0): # set default padding index as 0
        super(Embedder, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
    def forward(self, x):
        return self.embed(x)


# Positional Encoder
'''
d_model = dimension of Embedding vector
max_seq_len = maximum sequence length
dropout = dropout rate
'''
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # From tensorflow-applied version
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
        # PE_(pos, 0<= i < d_model/2) = sin(pos/10000 **(i / (d_model/2 - 1)))
        # PE_(pos, d_model/2 <= i < d_model) = cos(pos/10000 ** ((i - d_model/2)/(d_model/2 - 1)))
        d = d_model // 2.0 # float value
        pos = torch.arange(0.0, max_seq_len).unsqueeze(1)
        dim = torch.arange(0.0, d), torch.arange(d, d_model)
        term = pos / 10000 ** (dim[0]/(d-1)), pos / 10000 ** ((dim[1]-d)/(d-1))
        pe = torch.cat(term, dim=-1).unsqueeze(0)

        # keep the encoding values as constant (no gradient, not trainable)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1) # get sequence size from batch x sequence x d_model
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        x = x + pe
        return self.dropout(x)


'''
[Older]
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # create 'pe' matrix with values dependent in pos and dim
        pe = torch.zeros(max_seq_len, d_model) # max_seq_len x d_model
        
        # fill the encoding values
        # PE_(pos, 2i) = sin(pos/10000 ** (2i/d_model))
        # PE_(pos, 2i+1) = cos(pos/10000 ** (2i/d_model))
        pos = torch.arange(0.0, max_seq_len).unsqueeze(1)
        dim = torch.arange(0.0, d_model, 2) 
        term = pos / 10000 ** (dim/d_model)
        pe[:, 0::2] = torch.sin(term) # sin for even(2i) indices.
        pe[:, 1::2] = torch.cos(term) # cos for odd(2i+1) indices
        
        # add dimension 0 as a batch dimension
        pe = pe.unsqueeze(0) 
        
        # keep the encoding values as constant (no gradient, not trainable)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1) # get sequence size from batch x sequence x d_model
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        x = x + pe
        return self.dropout(x)
'''
#-----------------------------------------------------------------------------#


# Masking --------------------------------------------------------------------#
'''
* Two different types of masking are required: 1) Zeroing out '<pad>' from attention weights (encoder & decoder)
                                               2) Forcing 'the decoder' to use information of 'past positions' only
'''
# get padding mask : initial embedding vectors -> mask vectors (cuda -> cuda)
def get_mask(embedding, mask=True): # src_mask for source, tgt_mask for target returns None if mask = False 
    if mask:
        batch_size = embedding.size(0)
        return 1-(torch.sum(embedding, dim=-1)==0).view(batch_size, 1, 1, -1).float() # batch x 1 x 1 x sequence
    else: return None

# padding mask : mask vectors * attention vectors -> masked attention vectors (cuda -> cuda) 
def padding_mask(mask, attn_vec):
    return attn_vec * mask 

# subsequent mask : attention vectors -> masked attention vectors. (cuda -> cuda)
def subsequent_mask(attn_vec):
    return torch.tril(attn_vec, diagonal=0)
#-----------------------------------------------------------------------------#


# Attention ------------------------------------------------------------------#
'''
                    Q               K               V
                    |               |               |
              [h linear Qi]   [h linear Ki]   [h linear Vi]──────┐
                    |               |               |            |
          ┬      [ dot product & scale ]            |            |
          |                 |                       |            |
          |             [masking]                   |        Multi-head
      Attention             |                       |         Attention
          |             [softmax]                   | (h parallel calculations)
          |                 |                       |            |
          ┴                 └────[dot product]──────┘            |      
                                      |                          |                   
                          [concatenate h multi-heads]────────────┘
                                      |
                                   [linear]
'''
# Multi-headed Attention
class MultiHeadedAttention(nn.Module):
    '''
    * In order to divide Q, K, V into h groups, product (Q, K, V) with h different (W_Qi, W_Ki, W_Vi)s 
      which have (d_model, d_model/h(=d_k)) dimensions. (i = {1, ..., h})
    * Here, in order to obtain h different Q, K, V groups in a simpler way, we multiply Q, K, V with 
      a single W_Q, W_K, W_V which have (d_model x d_model) dimensions, and divide the results into 
      h groups. (batch, sequence, d_model) -> (batch, sequence, h, d_k) -> (batch, h, sequence, d_k)
    '''
    def __init__(self, heads, d_model, dropout=0.1, sequential=False):
        super(MultiHeadedAttention, self).__init__()
        # initialize variables
        self.h = heads
        self.d_model = d_model
        self.d_k = d_model // heads # create d_k (=projected dimension)
        self.sequential = sequential
        assert d_model % heads == 0, 'Number of heads must be a proper divisor of model dimension.'
        
        # initialize linear projection and dropout layers
        self.Q_linear = nn.Linear(d_model, d_model, bias=False)
        self.K_linear = nn.Linear(d_model, d_model, bias=False)
        self.V_linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # linear projection into (batch, sequence, h, d_k)
        Q = self.Q_linear(Q).view(batch_size, -1, self.h, self.d_k)
        K = self.K_linear(K).view(batch_size, -1, self.h, self.d_k)
        V = self.V_linear(V).view(batch_size, -1, self.h, self.d_k)
        
        # divide into h groups by transformation (batch, h, sequence, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # calculate attention-applied values (batch, h, sequence, d_k)
        attn_values = attention(Q, K, V, self.d_k, mask, self.sequential, self.dropout)
        
        # concatenate
        concat = attn_values.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # apply the last linear parameters
        output = self.out(concat)
        return output   


# Single-head scaled-dot-product attention
def attention(Q, K, V, d_k, mask=None, sequential=False, dropout=None):
    # dot product Q, K
    attn_vec = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k) # dot product - scale
    # mask
    if mask is not None:
        attn_vec = padding_mask(mask, attn_vec)
    # subsequent mask
    if sequential:
        attn_vec = subsequent_mask(attn_vec)
    # get attention weights
    attn_vec = attn_vec.masked_fill(attn_vec == 0, -np.inf) # perfectly zero-out
    scores = F.softmax(attn_vec, -1)
    # dropout
    if dropout is not None:
        scores = dropout(scores)
    # apply weights to V
    return torch.matmul(scores, V) # batch, h, sequence, d_k
#-----------------------------------------------------------------------------#


# Feed-Forward Network -------------------------------------------------------#
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
                
        # define linear calculations
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
#-----------------------------------------------------------------------------#
       

# Layer Normalization --------------------------------------------------------#
class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon = 1e-6):
        super(LayerNorm, self).__init__()
        self.size = d_model
        
        # create two learnable parameters to calibrate normalization
        self.alpha = nn.Parameter(torch.ones(self.size)) # initialize scale parameter 'alpha'
        self.bias = nn.Parameter(torch.zeros(self.size)) # initialize shift parameter 'bias'
        self.eps = epsilon
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
#-----------------------------------------------------------------------------#
        
    
# Encoder and Decoder Layers -------------------------------------------------#
'''
                        [1 Encoder Layer]  

                      layer normalization
                                |
                    1 multi-head-attention
                                |
                      residual connection
                                |
                      layer normalization
                                |
                         1 feed forward
                                |
                      residual connection
                                
'''        
# Single Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.attn = MultiHeadedAttention(heads, d_model, dropout=dropout, sequential=False)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # multi-head attention - residual connection
        x2 = self.norm1(x)
        x2 = self.dropout1(self.attn(x2, x2, x2, mask))
        x = x + x2
        
        # feed-forward - residual connection
        x2 = self.norm2(x)
        x2 = self.dropout2(self.ff(x2))
        x = x + x2
        return x
    
'''
                        [1 Decoder Layer]  

                      layer normalization                  ┬
                                |                          |
                    1 multi-head-attention                 | m-h attention 1  
                                |                          |
                      residual connection                  ┴
                                |
                      layer normalization                  ┬
                                |                          |
            1 multi-head-attention with encoded data       | m-h attention 2
                                |                          |
                      residual connection                  ┴
                                |                          
                      layer normalization                  ┬
                                |                          |
                         1 feed forward                    | feed-forward
                                |                          | 
                      residual connection                  ┴
  
'''
# Single Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.attn1 = MultiHeadedAttention(heads, d_model, dropout=dropout, sequential=True)
        self.attn2 = MultiHeadedAttention(heads, d_model, dropout=dropout, sequential=False)
        self.ff = FeedForward(d_model, dropout=dropout)
    
    def forward(self, x, memory, src_mask=None, trg_mask=None):
        # Multi-headed attention 1
        x2 = self.norm1(x)
        x2 = self.dropout1(self.attn1(x2, x2, x2, mask=trg_mask))
        x = x + x2
        # Multi-headed attention 2
        x2 = self.norm2(x)
        x2 = self.dropout2(self.attn2(x2, memory, memory, mask=src_mask))
        x = x + x2
        # Feed-forward
        x2 = self.norm3(x)
        x2 = self.dropout3(self.ff(x2))
        x = x + x2
        return x

    
# Layer duplicating function (layer -> a list of duplicated layers)
def DuplicateLayer(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# Encoder
class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, padding_idx=0, max_seq_len=200, dropout=0.1):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = DuplicateLayer(EncoderLayer(heads, d_model, dropout=dropout), N)
        self.norm = LayerNorm(d_model)
    
    def forward(self, x, src_mask):
        # loop through encoder layers
        for i in range(self.N):
            x = self.layers[i](x, src_mask)
        return self.norm(x)
 
    
# Decoder
class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, padding_idx=0, max_seq_len=200, dropout=0.1):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = DuplicateLayer(DecoderLayer(heads, d_model, dropout=dropout), N)
        self.norm = LayerNorm(d_model)
        
    def forward(self, x, memory, src_mask, trg_mask):
        # loop through decoder layers
        for i in range(self.N):
            x = self.layers[i](x, memory, src_mask, trg_mask)
        return self.norm(x)
#-----------------------------------------------------------------------------#


# Transformer ----------------------------------------------------------------#
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, N, heads, padding_idx=0, max_seq_len=200, dropout=0.1):
        super(Transformer, self).__init__()
        self.embed1 = Embedder(src_vocab_size, d_model, padding_idx=padding_idx)
        self.embed2 = Embedder(trg_vocab_size, d_model, padding_idx=padding_idx)
        self.pe1 = PositionalEncoder(d_model, max_seq_len=max_seq_len, dropout=dropout)
        self.pe2 = PositionalEncoder(d_model, max_seq_len=max_seq_len, dropout=dropout)
        self.encoder = Encoder(d_model, N, heads, padding_idx, max_seq_len, dropout)
        self.decoder = Decoder(d_model, N, heads, padding_idx, max_seq_len, dropout)
        self.out = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg, mask=True):
        # embed source and target inputs
        src = self.embed1(src)
        trg = self.embed2(trg)
        # get masks
        src_mask = get_mask(src, mask)
        trg_mask = get_mask(trg, mask)
        # positional encoding
        src = self.pe1(src)
        trg = self.pe2(trg)
        # encoding 
        memory = self.encoder(src, src_mask)
        output = self.decoder(trg, memory, src_mask, trg_mask)
        output = self.out(output)
        return output
#-----------------------------------------------------------------------------#



















