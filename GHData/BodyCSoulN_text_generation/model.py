import torch
from torch import nn
from torch.nn import functional as F
from data_preprocess import trans_dim

class RNNModel(nn.Module):
    """仅考虑单向的RNN，双向RNN不适合进行预测"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.rnn_layer = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn_layer.hidden_size
        self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        
    def forward(self, X, state):
        X = F.one_hot(X.T.long(), self.vocab_size).type(torch.float32)
        Y, state = self.rnn_layer(X, state)
        outputs = self.linear(Y.reshape((-1, Y.shape[-1])))
        return outputs, state

    def forward_0(self, X, state):
        X = F.one_hot(X.T.long(), self.vocab_size).type(torch.float32)
        state = trans_dim(state)
        Y, state = self.rnn_layer(X, state)
        state = trans_dim(state)
        outputs = self.linear(Y.reshape((-1, Y.shape[-1])))
        return outputs, state

    def forward_1(self, X, state):
        X = F.one_hot(X.long(), self.vocab_size).type(torch.float32)
        state = state.contiguous()
        Y, state = self.rnn_layer(X, state)
        state = state.contiguous()
        outputs = self.linear(Y.reshape((-1, Y.shape[-1]))).T
        return outputs, state
    
    def begin_state(self, batch_size, device):
        if not isinstance(self.rnn_layer, nn.LSTM):
            return torch.zeros(size=(self.rnn_layer.num_layers, batch_size, self.num_hiddens), device=device)
        else:
            return (torch.zeros(size=(self.rnn_layer.num_layers, batch_size, self.num_hiddens), device=device), 
                   torch.zeros(size=(self.rnn_layer.num_layers, batch_size, self.num_hiddens), device=device))