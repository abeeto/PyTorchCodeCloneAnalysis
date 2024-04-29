import torch
from torch import nn

# initializing the base rnn for 2D compute...

class RNNCell(nn.Module):
    
    def __init__(self, n_x, n_a, n_y):
        super(RNNCell, self).__init__()
        
        # set shapes..
        
        self.n_x = n_x
        self.n_a = n_a
        self.n_y = n_y
        self.Wax = torch.rand((self.n_a, self.n_x))
        self.Waa = torch.rand((self.n_a, self.n_a))
        self.Wya = torch.rand((self.n_y, self.n_a))
        self.ba = torch.rand((self.n_a, 1))
        self.by = torch.rand((self.n_y, 1))
        self.wa = torch.cat([self.Waa, self.Wax], dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
    def forward(self, xt, a_prev):
        self.ax = torch.cat([a_prev, xt], dim=0)
        self.a_next = self.tanh(self.wa @ self.ax + self.ba)
        self.yt = self.softmax(self.Wya @ self.a_next + self.by)
        return self.a_next, self.yt

 # main RNN class for compute 3D inputs.   
 
class RNN(nn.Module):
    def __init__(self, n_x, n_a, t_x, m):
        super().__init__()
        '''
        n_x -> input size
        m -> batch_size
        T_x -> no. of time steps
        '''
        self.n_x = n_x
        self.t_x = t_x
        self.n_a = n_a
        self.m = m
        self.rnncell = RNNCell(n_x, n_a, n_x)
        self.a_next = torch.zeros(n_a, m, t_x)
        self.y_pred = torch.zeros(n_x, m, t_x)
    def forward(self, x, a0):
        for i in range(self.t_x):
            a0, yt = self.rnncell(x[:,:,i], a0)
            self.a_next[:,:,i] = a0
            self.y_pred[:,:,i] = yt
        return self.a_next, self.y_pred

# you could test by running following code. 

# rnn = RNN(500, 5, 10, 2)
# x = torch.rand((500, 2, 10))
# a0 = torch.zeros(5, 2)

# a, y = rnn(x, a0)
# print(a.shape, y.shape)
