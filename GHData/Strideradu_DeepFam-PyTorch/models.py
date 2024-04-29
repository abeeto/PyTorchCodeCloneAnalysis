import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PepCNN(nn.Module):

    def __init__(self, num_class=1000, num_token=21, seq_len=1000, kernel_nums=[256, 256, 256, 256, 256, 256, 256, 256],
                 kernel_sizes=[8, 12, 16, 20, 24, 28, 32, 36], dropout=0.5, num_fc=512):
        super(PepCNN, self).__init__()

        self.num_token = num_token
        self.seq_len = seq_len
        self.num_class = num_class
        self.channle_in = 1
        self.kernel_nums = kernel_nums
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout

        # self.embed = nn.Embedding(self.num_token, self.seq_len)
        # self.convs1 = [nn.Conv2d(Ci, self.kernel_num, (kernel_size, D)) for kernel_size in self.kernel_size]
        self.convs1 = nn.ModuleList(
            [ConvBNRelu2d(self.channle_in, self.kernel_nums[i], kernel_size= (kernel_size, self.num_token)) for i, kernel_size in
             enumerate(self.kernel_sizes)])
        '''
        self.conv13 = nn.Conv2d(Ci, self.kernel_num, (3, D))
        self.conv14 = nn.Conv2d(Ci, self.kernel_num, (4, D))
        self.conv15 = nn.Conv2d(Ci, self.kernel_num, (5, D))
        '''
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(sum(self.kernel_nums), num_fc)
        self.fc2 = nn.Linear(num_fc, self.num_class)

    def forward(self, x):
        # x = self.embed(x)  # (N, W, D)

        # if self.args.static:
        #     x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [conv(x).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.fc1(x)  # (N, C)
        logit = self.fc2(x)
        return logit

class ConvBNRelu2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBNRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class PepCNN_v2(nn.Module):

    def __init__(self, num_class=1000, num_token=21, seq_len=1000, kernel_nums=[256, 256, 256, 256, 256, 256, 256, 256],
                 kernel_sizes=[8, 12, 16, 20, 24, 28, 32, 36], dropout=0.5, num_fc=512):
        super(PepCNN_v2, self).__init__()

        self.num_token = num_token
        self.seq_len = seq_len
        self.num_class = num_class
        self.channle_in = 1
        self.kernel_nums = kernel_nums
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout

        # self.embed = nn.Embedding(self.num_token, self.seq_len)
        # self.convs1 = [nn.Conv2d(Ci, self.kernel_num, (kernel_size, D)) for kernel_size in self.kernel_size]
        self.dropout_2d = nn.Dropout2d(0.05)
        self.convs1 = nn.ModuleList(
            [nn.Conv2d(self.channle_in, self.kernel_nums[i], (kernel_size, self.num_token)) for i, kernel_size in
             enumerate(self.kernel_sizes)])
        '''
        self.conv13 = nn.Conv2d(Ci, self.kernel_num, (3, D))
        self.conv14 = nn.Conv2d(Ci, self.kernel_num, (4, D))
        self.conv15 = nn.Conv2d(Ci, self.kernel_num, (5, D))
        '''
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(sum(self.kernel_nums), num_fc)
        self.fc2 = nn.Linear(num_fc, self.num_class)

    def forward(self, x):
        # x = self.embed(x)  # (N, W, D)

        # if self.args.static:
        #     x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = self.dropout_2d(x)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        x = self.fc1(x)  # (N, C)
        logit = self.fc2(x)
        return logit
