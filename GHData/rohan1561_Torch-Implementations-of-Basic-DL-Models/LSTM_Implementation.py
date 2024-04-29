import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.autograd as autograd
from torch.autograd import Variable
import math

class LSTMCell(nn.Module):
    """
    LSTM cell implementation
    Given an input x at time step t, and hidden and cell states: hidden = (h_(t-1), c_(t-1)),
    this is an LSTM unit to compute and return (h_t, c_t)
    """
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        # TODO:
 	self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_hid = Parameter(torch.Tensor(input_size, hidden_size*4))
        self.hid_hid = Parameter(torch.Tensor(hidden_size, hidden_size*4))
        self.bias = torch.zeros(hidden_size*4, requires_grad=True).cuda()
        self.init_w()
         
    def init_w(self):
        '''
        This function initializes the weights and biases. The weights are 
        initialized as Xavier Uniform (for faster convergence). Refer:
        'Understanding the difficulty of training deep feedforward neural
        networks` - Glorot, X. & Bengio, Y. (2010)'

        The biases are initialized as zeros to capture long term dependencies
        better
        '''
        for p in self.parameters():
            if p.data.ndimension() >=2:
                nn.init.xavier_uniform_(p.data)

    def forward(self, x, hidden):
        # TODO:
        H = self.hidden_size
        h_n, c_n = hidden
        # Carry out matrix multiplications
        all_gates = torch.matmul(x, self.input_hid) +\
                torch.matmul(h_n, self.hid_hid) + self.bias #(b,4h)

        # Retrieve the gates (input, forget, candidate, output)
        i_n, f_n, cg_n, o_n = (torch.sigmoid(all_gates[:, :H]),\
                torch.sigmoid(all_gates[:, H:H*2]),\
                torch.tanh(all_gates[:, H*2:H*3]),\
                torch.sigmoid(all_gates[:, H*3:]))

        # New states
        c_n = i_n*cg_n + f_n*c_n # New cell state (b,h)
        h_n = o_n*torch.tanh(c_n) # New hidden state(b,h)

        hidden = (h_n, c_n)
    
        return hidden

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size # dimension of hidden states
        self.lstmcell = LSTMCell(input_size, hidden_size)

    def forward(self, x, states):
        h0, c0 = states
        outs = []
        cn = c0[0, :, :] # (b, h)
        hn = h0[0, :, :] # (b, h)
        for seq in range(x.size(1)):
            hn, cn = self.lstmcell(x[:, seq, :], (hn, cn))
            outs.append(hn)
        out = torch.stack(outs, 1) #(b,seq,h)
        states = (hn.unsqueeze(0), cn.unsqueeze(0)) #(1,b,h)
        return out, states

