# -*- coding: utf-8 -*-
import sys

import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T
import torch.optim as optim

# if gpu is to be used
use_cuda = torch.cuda.is_available()
if use_cuda:
    print >> sys.stderr, "GPU is available!"
else:
    print >> sys.stderr, "GPU is not available!"
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

torch.manual_seed(0)

class Network(nn.Module):

    def __init__(self, pair_feature_num, ana_feature_num, word_num, span_num, hidden_num, embedding_size, embedding_dimention, embedding_matrix, label_num=1):

        super(Network,self).__init__()

        self.embedding_layer = nn.Embedding(embedding_size,embedding_dimention)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))
        #self.embedding_layer.weight.requires_grad=False

        self.mention_word_layer = nn.Linear(word_num,hidden_num)
        self.mention_span_layer = nn.Linear(span_num,hidden_num)
        self.candi_word_layer = nn.Linear(word_num,hidden_num)
        self.candi_span_layer = nn.Linear(span_num,hidden_num)
        self.pair_feature_layer = nn.Linear(pair_feature_num,hidden_num)

        self.hidden_layer_1 = nn.Linear(hidden_num, hidden_num/2)
        self.hidden_layer_2 = nn.Linear(hidden_num/2, hidden_num/2)
        self.output_layer = nn.Linear(hidden_num/2, 1)

        self.ana_word_layer = nn.Linear(word_num,hidden_num)
        self.ana_span_layer = nn.Linear(span_num,hidden_num)
        self.ana_feature_layer = nn.Linear(ana_feature_num,hidden_num)

        self.ana_hidden_layer_1 = nn.Linear(hidden_num, hidden_num/2)
        self.ana_hidden_layer_2 = nn.Linear(hidden_num/2, hidden_num/2)
        self.ana_output_layer = nn.Linear(hidden_num/2, 1)

        self.activate = F.relu
        self.softmax_layer = nn.Softmax()

    def forward(self, word_embedding_rep_dimention, ana_word_index, ana_span, ana_feature, mention_word_index, mention_span, candi_word_index, candi_span, pair_feature, dropout=0.0):

        dropout_layer = nn.Dropout(dropout)
        
        self.parameters = []
    
        mention_word_embedding = self.embedding_layer(mention_word_index).view(-1,word_embedding_rep_dimention)
        mention_word_embedding = dropout_layer(mention_word_embedding)
        mention_embedding_rep = self.mention_word_layer(mention_word_embedding)
        mention_span_rep = self.mention_span_layer(mention_span)

        self.parameters.append(self.mention_span_layer.parameters())

        candi_word_embedding = self.embedding_layer(candi_word_index).view(-1,word_embedding_rep_dimention)
        candi_word_embedding = dropout_layer(candi_word_embedding)
        candi_embedding_rep = self.candi_word_layer(candi_word_embedding)
        candi_span_rep = self.candi_span_layer(candi_span)

        pair_feature_rep = self.pair_feature_layer(pair_feature)

        inpt = mention_embedding_rep + mention_span_rep + candi_embedding_rep + candi_span_rep + pair_feature_rep
        inpt = self.activate(inpt)

        x = dropout_layer(inpt)
        x = self.hidden_layer_1(x)
        x = self.activate(x)

        x = dropout_layer(x)
        x = self.hidden_layer_2(x)
        x = self.activate(x)

        x = dropout_layer(x)
        x = torch.transpose(self.output_layer(x),0,1)

        ## deal with anaphora
        #ana_word_embedding = self.embedding_layer(ana_word_index).view(-1,word_embedding_rep_dimention)
        #ana_embedding_rep = self.ana_word_layer(ana_word_embedding)
        #ana_span_rep = self.ana_span_layer(ana_span)

        ana_embedding_rep = dropout_layer(self.ana_word_layer(mention_word_embedding))
        ana_span_rep = self.ana_span_layer(mention_span)
        ana_feature_rep = self.ana_feature_layer(ana_feature)

        ana_input = ana_embedding_rep + ana_span_rep + ana_feature_rep
        ana_input = self.activate(ana_input)

        xs = dropout_layer(ana_input)
        xs = self.ana_hidden_layer_1(xs)
        xs = self.activate(xs)

        xs = dropout_layer(xs)
        xs = self.ana_hidden_layer_2(xs)
        xs = self.activate(xs)

        xs = dropout_layer(xs)
        xs = self.ana_output_layer(xs)

        x = torch.cat((xs,x),1)

        output = F.sigmoid(x)
        softmax_out = self.softmax_layer(x)

        return output,softmax_out

    def forward_all_pair(self, word_embedding_rep_dimention, mention_word_index, mention_span, candi_word_index, candi_span, pair_feature, anaphors, antecedents, dropout=0.0):

        dropout_layer = nn.Dropout(dropout)
        #factor = 1.0/(1.0-dropout)
        factor = 1.0

        mention_word_embedding = self.embedding_layer(mention_word_index).view(-1,word_embedding_rep_dimention)
        mention_word_embedding = dropout_layer(mention_word_embedding[anaphors])
        mention_embedding_rep = self.mention_word_layer(mention_word_embedding)

        mention_span_rep = self.mention_span_layer(mention_span)[anaphors]

        candi_word_embedding = self.embedding_layer(candi_word_index).view(-1,word_embedding_rep_dimention)
        candi_word_embedding = dropout_layer(candi_word_embedding[antecedents])
        candi_embedding_rep = self.candi_word_layer(candi_word_embedding)

        candi_span_rep = self.candi_span_layer(candi_span)[antecedents]

        pair_feature_rep = self.pair_feature_layer(pair_feature)

        inpt = mention_embedding_rep + mention_span_rep + candi_embedding_rep + candi_span_rep + pair_feature_rep
        inpt = self.activate(inpt)

        x = dropout_layer(inpt)
        x = self.hidden_layer_1(x)
        x = self.activate(x)

        x = dropout_layer(x)
        x = self.hidden_layer_2(x)
        x = self.activate(x)

        x = dropout_layer(x)
        x = torch.transpose(self.output_layer(x),0,1)

        output = F.sigmoid(x)

        return output,x

    def forward_anaphoricity(self, word_embedding_rep_dimention, word_index, span, feature, dropout=0.0):

        dropout_layer = nn.Dropout(dropout)
        #factor = 1.0/(1.0-dropout)
        factor = 1.0

        anaphoricity_word_embedding = self.embedding_layer(word_index).view(-1,word_embedding_rep_dimention)
        anaphoricity_word_embedding = dropout_layer(anaphoricity_word_embedding)
        anaphoricity_embedding_rep = self.ana_word_layer(anaphoricity_word_embedding)

        anaphoricity_span_rep = self.ana_span_layer(span)
        anaphoricity_feature_rep = self.ana_feature_layer(feature)

        ana_input = anaphoricity_embedding_rep + anaphoricity_span_rep + anaphoricity_feature_rep
        ana_input = self.activate(ana_input)

        xs = dropout_layer(ana_input)
        xs = self.ana_hidden_layer_1(xs)
        xs = self.activate(xs)

        xs = dropout_layer(xs)
        xs = self.ana_hidden_layer_2(xs)
        xs = self.activate(xs)

        xs = dropout_layer(xs)
        xs = self.ana_output_layer(xs)
        x = torch.transpose(xs,0,1)

        output = 1-F.sigmoid(x) # output is the probability of anaphoricty. if x is big, means a higher probability of un-anaphoric, thusthere is 1-sigmoid(x)

        return output,x

    def forward_top_pair(self, word_embedding_rep_dimention, mention_word_index, mention_span, candi_word_index, candi_span, pair_feature, anaphors, antecedents, reindex, starts, ends, dropout=0.0):

        dropout_layer = nn.Dropout(dropout)
        #factor = 1.0/(1.0-dropout)
        factor = 1.0

        mention_word_embedding = self.embedding_layer(mention_word_index).view(-1,word_embedding_rep_dimention)
        mention_word_embedding = dropout_layer(mention_word_embedding[anaphors])
        mention_embedding_rep = self.mention_word_layer(mention_word_embedding)

        mention_span_rep = self.mention_span_layer(mention_span)[anaphors]

        candi_word_embedding = self.embedding_layer(candi_word_index).view(-1,word_embedding_rep_dimention)
        candi_word_embedding = dropout_layer(candi_word_embedding[antecedents])
        candi_embedding_rep = self.candi_word_layer(candi_word_embedding)

        candi_span_rep = self.candi_span_layer(candi_span)[antecedents]

        pair_feature_rep = self.pair_feature_layer(pair_feature)

        inpt = mention_embedding_rep + mention_span_rep + candi_embedding_rep + candi_span_rep + pair_feature_rep
        inpt = self.activate(inpt)

        x = dropout_layer(inpt)
        x = self.hidden_layer_1(x)
        x = self.activate(x)

        x = dropout_layer(x)
        x = self.hidden_layer_2(x)
        x = self.activate(x)

        x = dropout_layer(x)
        x = self.output_layer(x)

        output = F.sigmoid(x)
        output_reindex = output[reindex]

        max_output = []
        for i in range(len(starts)):
            max_output.append(torch.max(output_reindex[starts[i].data[0]:ends[i].data[0]]))

        return torch.cat(max_output),output[reindex]


def main():
    #def __init__(self, feature_num, ana_feature_num, word_num, span_num, hidden_num, embedding_size, embedding_dimention, embedding_matrix, label_num=1):

    eb = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6]])

    model = Network(2,2,12,4,6,6,3,eb).cuda()
    
    print model

    #def forward(self, word_embedding_rep_dimention, ana_word_index, ana_span, ana_feature, mention_word_index, mention_span, candi_word_index, candi_span, pair_feature, dropout=0.0):

    ana_word_index = autograd.Variable(torch.cuda.LongTensor([1,2,3,5]))
    ana_span = autograd.Variable(torch.cuda.FloatTensor([[1,1,1,1]]))
    ana_feature = autograd.Variable(torch.cuda.FloatTensor([[8,8]]))
    
    mention_word_index = autograd.Variable(torch.cuda.LongTensor([4,4,4,5]))
    mention_span = autograd.Variable(torch.cuda.FloatTensor([[1,1,5,9]])) 

    candi_word_index = autograd.Variable(torch.cuda.LongTensor([[4,4,4,5],[1,1,5,3]]))
    candi_span = autograd.Variable(torch.cuda.FloatTensor([[1,1,5,9],[9,9,4,5]])) 

    pair_feature = autograd.Variable(torch.cuda.FloatTensor([[2,8],[6,5]]))

    print model.forward(12, ana_word_index, ana_span, ana_feature, mention_word_index, mention_span, candi_word_index, candi_span, pair_feature)
    
    #loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    #print model(autograd.Variable(input_vector))

    target = torch.cuda.FloatTensor([1,0,1])

    for i in range(10):
        optimizer.zero_grad()
        output = model.forward(12, ana_word_index, ana_span, ana_feature, mention_word_index, mention_span, candi_word_index, candi_span, pair_feature)
        loss = F.binary_cross_entropy(output,autograd.Variable(target))
        loss.backward()
        optimizer.step()    # Does the update

    print model.forward(12, ana_word_index, ana_span, ana_feature, mention_word_index, mention_span, candi_word_index, candi_span, pair_feature)

if __name__ == "__main__":
    main()
