# coding: utf-8

import torch
from torch.utils.data import DataLoader, Dataset
# import torch.utils.data.DataLoader as DataLoader
import codecs
from torch.autograd import Variable
import torch.nn as nn
import time
import torch.nn.init
import torch.nn.functional as F

torch.manual_seed(2)

time1 = time.time()
class My_data(Dataset):
    def __init__(self, file):
        self.file = codecs.open(file, 'rb', 'utf-8')
        self.num = int(self.file.readline().strip())
        self.lis = []
        for i in self.file:
            self.lis.append(int(i.strip()))

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        if item % 2 == 0:
            return [torch.FloatTensor([[self.lis[item], 99]]), 1]
        else:
            return [torch.FloatTensor([[self.lis[item], 79]]), 1,0]
# class Batch_gen(object):
#     def __init__(self, file_name, batch_size):
#         self.batch_size = batch_size
#         self.file = codecs.open(file_name, 'rb')
#         self.pairs = []
#         for line in self.file:
#             if line.strip():
#                 str_pair = line.strip().split('\t')
#                 str_seq = str_pair[0].split(' ')
#                 seq = []
#                 for i in str_seq:
#                     seq.append(int(i))
#                 int_pair = [seq, int(str_pair[1])]
#                 self.pairs.append(int_pair)
#         self.cursor = 0
#
#     def __iter__(self):
#         return self
#
#     def next(self):
#         if self.cursor < len(self.pairs):
#             result = self.pairs[self.cursor:self.cursor+self.batch_size]
#             self.cursor += self.batch_size
#             seq_pad = []
#             label_pad
#             biggest = 0
#             for pair in result:
#                 length = len(pair[0])
#                 lengths.append(length)
#                 if length > biggest:
#                     biggest = length
#             for pair in result:
#                 if len(pair)
#
#             return result
#         else:
#             raise StopIteration("out of list")
# data = Batch_gen('data', 4)
# for i, batch in enumerate(data):
#     print i, batch






# data = My_data('data')
# myLoader = DataLoader(data, batch_size=2, shuffle=True)
# for i, batch in enumerate(myLoader):
#     print i, batch

# import torch.nn.functional as F
#
# input = Variable(torch.rand(1, 1, 10)) # 1D (N, C, L)
# input_2d = input.unsqueeze(2) # add a fake height
# print(input_2d)
# print(F.pad(input_2d, (2, 0, 0, 0)).view(1, 1, -1)) # left padding (and remove height)
# F.pad(input_2d, (0, 2, 0, 0)).view(1, 1, -1) # right padding (and remove height)




batch = [Variable(torch.LongTensor([0,1,2,3])), Variable(torch.LongTensor([0,1,2,3,4]))]
label = [Variable(torch.LongTensor([0,1])), Variable(torch.LongTensor([1,0]))]


class EncoderRNN(nn.Module):
    def __init__(self, voc_size, emb_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.emb_size = emb_size

        # self.embedding = nn.Embedding(voc_size, emb_size)
        setattr(self, 'embedding', nn.Embedding(voc_size, emb_size))

        self.gru = nn.GRU(emb_size, hidden_size, num_layers=n_layers)
        # self.l1 = nn.Linear(3, 2)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(-1, 1, self.emb_size)
        hid = hidden
        outputs = []
        for i in range(embedded.size()[0]):
            input = embedded[i].view(1,1,-1)
            output, hid = self.gru(input, hid)
            outputs.append(output[0][0])
        return outputs


        # output = embedded
        # for i in range(self.n_layers):
        #     output, hidden = self.gru(output, hidden)
        # return output, hidden

        # self.emb = nn.Embedding(3, 2)
        # self.emb.weight.data.copy_(torch.FloatTensor([[1,0],[0,1],[0,0]]))
        # print self.emb.weight
        # return self.emb(Variable(torch.LongTensor([0])))
# a = EncoderRNN(2,3,4,5)
# print a(1,1)
# print a.emb.weight
# print(list(a.modules()))

# class Dec

# a = EncoderRNN(5, 3, 4, 3)
# # print a(batch[0], Variable(torch.FloatTensor([[[0.1,0.1,0,1]],[[0.1,0.1,0,1]],[[0.1,0.1,0,1]]])))
# # print a.embedding.weight
# a = Variable(torch.FloatTensor([1,2]), requires_grad=True)
# b = a * 2
# b.detach_()
# b[0] = 100
# print b.requires_grad
# x = Variable(torch.FloatTensor([1,2]), requires_grad=True)
# c = b *3 + x
# print c
# # d[0] = 100
# # print b
# c.sum().backward()
# print a.grad
# # embedding = nn.Embedding(10, 15)
# # embedded = embedding(Variable(torch.LongTensor([0,1,2,3]))).view(-1, 1, 15)
# # print(embedded[0])
# beam = [{'paths':[(),(),(),()] , 'tails':[] },{'paths':[(),(),(),()] , 'tails':[]}]
# print beam
#
# a = Variable(torch.FloatTensor([[1,2,3],[4,5,6]]))
# val, ind = a.topk(1,0)
# print val, ind
#
#
#
# w = Variable(torch.rand(3)).cuda(0)
# print w
# a = w.unsqueeze(1)
# print a
# # w[0] = 0
# print a
# torch.nn.init.orthogonal(w, 1)
# print w
# a = nn.GRU(1,2,3)
# print a.bias_hh_l2
# for name, param in a.named_parameters():
#     print name, param
# print w.cuda(0)
# x = torch.randn(4, 4)
# y = x.view(-1)
# print x
# print y
# y[0] = 1000
# print x, y
# print [0 for i in range(10)]
# import numpy
# seq = numpy.ones((3,2), dtype='int32')
# pad = numpy.zeros((2, seq.shape[1]), dtype='int32')
# pad_seq = numpy.vstack((seq, pad))
# a = numpy.array([0,1,2])
# print numpy.concatenate((pad_seq[numpy.newaxis, ...],pad_seq[numpy.newaxis,...]), axis=0)
# w = Variable(torch.LongTensor([0])).cuda(0)
# # z,_ = torch.max(w, 1)
# # z[0,0,0] = 100
# print w.data[0]

# print torch.cat((w,w), dim=-1)
# z[0,0] = 100
# # l = nn.Linear(2,4)
# # l.cuda(0)
# print w,z
# print w.size()[0]
class Multi(nn.Module):
    def __init__(self):
        super(Multi, self).__init__()
        # self.linear = nn.Linear(2,3)
        self.hidden = nn.Parameter(torch.ones(1,2), requires_grad=True)
        self.encoder_gru = nn.RNN(input_size=25, hidden_size=75,
                                  num_layers=1, bidirectional=True)

    def forward(self, input,hidden):
        # print '---', hidden, '---'
        input = input.transpose(0, 1).contiguous()
        hidden = hidden.transpose(0, 1).contiguous()
        gru_output, h_n = self.encoder_gru(input, hidden)
        return gru_output.transpose(0,1)  # (seq_len, batch, hidden_size * num_directions)

# data = Variable(torch.ones(100,50,25), requires_grad=True)
# cuda_data = data.cuda(3)
# model0 = Multi()
# a = Variable(torch.ones(100,2,75), requires_grad=True)
# b = a * 2
# a = a.transpose(0,1)
# a =

# model = nn.DataParallel(model0, device_ids=[3, 1])
# model.cuda(3)
# output = Variable(torch.FloatTensor(3,2,4)).cuda(3)
# loss = Variable(torch.FloatTensor(1)).cuda(3)
# output = model(data.cuda(3), b.cuda(3))
# loss = output.sum()
# for j in range(1000):
#     output = Variable(torch.FloatTensor(3, 2, 4)).cuda(3)
#     loss = Variable(torch.FloatTensor(1)).cuda(3)
#     for i in range(100):
#         output = model(data.cuda(3), b.cuda(3))
#         loss += output.sum()
#     print 'fi'
#     loss.backward()
# print data.data.topk(1)
def softmax(input, axis=1):
    input_size = input.size()

    trans_input = input.transpose(axis, len(input_size) - 1)
    trans_size = trans_input.size()

    input_2d = trans_input.contiguous().view(-1, trans_size[-1])

    soft_max_2d = F.softmax(input_2d)

    soft_max_nd = soft_max_2d.view(*trans_size)
    return soft_max_nd.transpose(axis, len(input_size) - 1)

# loss.backward()
# print output
# print data.grad
# a = torch.FloatTensor(6,2,5)
# b = torch.FloatTensor([[1,3]]).squeeze(0)
# print a.dot(b)
# for name, param in model0.named_parameters():
#     print name, param.grad
# # print torch.mm(cuda_data, nn.Parameter(torch.cuda.LongTensor(2,2)).type(torch.cuda.FloatTensor))
# zip()
# f_max = 0
# fl = [1,2,3,4,5,6,7,8,9,10,9,8,7,6,10,11]
# for i in range(1000):
#     f = fl[i]
#     if f >= f_max:
#         f_max = f
#         low_epoch = 0
#         print 'bigger'
#     else:
#         low_epoch += 1
#         print 'smaller'
#     if low_epoch >= 5:
#         break
# remove annoying characters
import re
chars = {
    '\xc2\x82' : ',',        # High code comma
    '\xc2\x84' : ',,',       # High code double comma
    '\xc2\x85' : '...',      # Tripple dot
    '\xc2\x88' : '^',        # High carat
    '\xc2\x91' : '\x27',     # Forward single quote
    '\xc2\x92' : '\x27',     # Reverse single quote
    '\xc2\x93' : '\x22',     # Forward double quote
    '\xc2\x94' : '\x22',     # Reverse double quote
    '\xc2\x95' : ' ',
    '\xc2\x96' : '-',        # High hyphen
    '\xc2\x97' : '--',       # Double hyphen
    '\xc2\x99' : ' ',
    '\xc2\xa0' : ' ',
    '\xc2\xa6' : '|',        # Split vertical bar
    '\xc2\xab' : '<<',       # Double less than
    '\xc2\xbb' : '>>',       # Double greater than
    '\xc2\xbc' : '1/4',      # one quarter
    '\xc2\xbd' : '1/2',      # one half
    '\xc2\xbe' : '3/4',      # three quarters
    '\xca\xbf' : '\x27',     # c-single quote
    '\xcc\xa8' : '',         # modifier - under curve
    '\xcc\xb1' : ''          # modifier - under line
}
def replace_chars(match):
    char = match.group(0)
    return chars[char]

emb_file = 'res_cmn/embedding_char.txt' #'em'#
voc = 'res_cmn/cmn_voc_char.txt' #'mem'#
# voc.replace()
# a = '(\xc2\x85) 0.067155'
# print a.decode('utf-8')
voc_fine = codecs.open(voc, mode='wb', encoding='utf-8')
with codecs.open(emb_file, mode='rb', encoding='utf-8') as f:
    lst = []
    for i, line in enumerate(f):
        line = line.strip()
        # if line.find('\xc2\x85'.decode('utf-8')) != -1:
        #     print i, line, line.find('\xc2\x85'.decode('utf-8'))
        #
        # voc_fine.write(line.replace('\xc2\x85'.decode('utf-8'), ''))
        # if line:
        #     voc_fine.write(line+'\n')
        # print type(line)

        if line:
            if i == 0:
                continue
                # parts = line.split(' ')
                # self.voc_size = int(parts[0]) + 8
                # self.emb_size = int(parts[1])
                # self.embedding_tensor = torch.FloatTensor(self.voc_size, self.emb_size)
            else:
                parts = line.split(' ')
                voc_fine.write(parts[0]+'\t'+str(i+2)+'\n')
    #     if line:
    #         lst.append(line.split('\t')[0])
    # words = set(lst)
    # for item in words:
    #     num = lst.count(item)
    #     if num > 1:
    #         print(u"the {} has found {}".format(item, num))

voc_fine.flush()
voc_fine.close()


