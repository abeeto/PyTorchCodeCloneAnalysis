import torch
import torch.nn as nn
from torch.autograd import Variable

class Classifier(nn.Module):
    def __init__(self,num_voca, emb_dim, hidden_dim,use_cuda):
        super(Classifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.use_cuda = use_cuda

        self.Embedding = nn.Embedding(num_voca,emb_dim)
        self.lstm = nn.LSTM(emb_dim,hidden_dim, batch_first=True,bidirectional=True,num_layers=2,dropout=0.5)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(hidden_dim*2,1)
        self.init_params()

    def forward(self,x):
        _ = self.Embedding(x)
        h0,c0 = self.init_hidden(x.size(0))
        _,(h,c) = self.lstm(_,(h0,c0))
        _ = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
        _ = self.linear(self.dropout(_))
        return _

    def init_hidden(self,batch_size):
        h = Variable(torch.zeros((2*2,batch_size,self.hidden_dim)))
        c = Variable(torch.zeros((2*2,batch_size,self.hidden_dim)))
        if self.use_cuda:
            h,c = h.cuda(),c.cuda()
        return h,c

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)