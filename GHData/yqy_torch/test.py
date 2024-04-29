#coding=utf8

import sys
import os
import json
import random
import numpy
import timeit

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T
import torch.optim as optim

from conf import *

import DataReader
import evaluation
import network
import Evaluate

import cPickle
sys.setrecursionlimit(1000000)

print >> sys.stderr, os.getpid()

if args.language == "en":
    pair_feature_dimention = 77
    mention_feature_dimention = 24
    span_dimention = 5*50
    embedding_dimention = 50
    embedding_size = 34275
    word_embedding_dimention = 9*50
else:
    pair_feature_dimention = 77
    mention_feature_dimention = 24
    span_dimention = 5*50
    embedding_dimention = 50
    embedding_size = 34275
    word_embedding_dimention = 9*50

torch.cuda.set_device(args.gpu)
 
def main():

    DIR = args.DIR
    embedding_file = args.embedding_dir

    network_file = "./model/model.pkl"
    if os.path.isfile(network_file):
        print >> sys.stderr,"Read model from ./model/model.pkl"
        network_model = torch.load(network_file)
    else:
        embedding_matrix = numpy.load(embedding_file)

        "Building torch model"
        network_model = network.Network(pair_feature_dimention,mention_feature_dimention,word_embedding_dimention,span_dimention,1000,embedding_size,embedding_dimention,embedding_matrix).cuda()
        print >> sys.stderr,"save model ..."
        torch.save(network_model,network_file)

    reduced=""
    if args.reduced == 1:
        reduced="_reduced"

    train_docs = DataReader.DataGnerater("train"+reduced)
    dev_docs = DataReader.DataGnerater("dev"+reduced)
    test_docs = DataReader.DataGnerater("test"+reduced)


    l2_lambda = 1e-5
    lr = 0.002
    dropout_rate = 0.5
    shuffle = True
    times = 0
    best_thres = 0.5

    model_save_dir = "./model/pretrain/"
   
    last_cost = 0.0
     
    for echo in range(30):

        start_time = timeit.default_timer()
        print "Pretrain Epoch:",echo

        optimizer = optim.RMSprop(network_model.parameters(), lr=lr, weight_decay=l2_lambda)

        cost_this_turn = 0.0

        pos_num = 0
        neg_num = 0
        inside_time = 0.0
    
        loss = None

        for data,doc_end in train_docs.generater(shuffle):
            ana_word_index,ana_span,ana_feature,candi_word_index,candi_span,pair_feature_array,target,mention_ids = data


            if len(pair_feature_array) >= 500:
                continue
            if len(target) == 0:
                continue
                

            mention_index = autograd.Variable(torch.from_numpy(ana_word_index).type(torch.cuda.LongTensor))
            mention_span = autograd.Variable(torch.from_numpy(ana_span).type(torch.cuda.FloatTensor))
            mention_feature = autograd.Variable(torch.from_numpy(ana_feature).type(torch.cuda.FloatTensor))
            candi_index = autograd.Variable(torch.from_numpy(candi_word_index).type(torch.cuda.LongTensor))
            candi_spans = autograd.Variable(torch.from_numpy(candi_span).type(torch.cuda.FloatTensor))
            pair_feature = autograd.Variable(torch.from_numpy(pair_feature_array).type(torch.cuda.FloatTensor))

            gold = [0] + target.tolist()
            if sum(target) == 0:
                neg_num += 1
                gold[0] = 1
            else:
                pos_num += 1

            inside_time_start = timeit.default_timer()

            lable = autograd.Variable(torch.cuda.FloatTensor([gold]))
            output,scores = network_model.forward(word_embedding_dimention,mention_index,mention_span,mention_feature,mention_index,mention_span,candi_index,candi_spans,pair_feature,dropout_rate)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy(output,lable)
            loss.backward()
            optimizer.step()
            inside_time += (timeit.default_timer()-inside_time_start)
            cost_this_turn += loss.data[0]


        end_time = timeit.default_timer()
        print >> sys.stderr, "PreTrain",echo,"Total cost:",cost_this_turn
        print >> sys.stderr, "PreTRAINING Use %.3f seconds"%(end_time-start_time)
        print >> sys.stderr, "Inside Use %.3f seconds"%(inside_time)
        print >> sys.stderr, "Neg:Pos",neg_num,pos_num
        print >> sys.stderr, "Learning Rate",lr

        if cost_this_turn > last_cost:
            lr = lr*0.7 
        last_cost = cost_this_turn

        print >> sys.stderr,"save model ..."

        best_thres = Evaluate.evaluate(network_model,dev_docs,best_thres)

if __name__ == "__main__":
    main()
