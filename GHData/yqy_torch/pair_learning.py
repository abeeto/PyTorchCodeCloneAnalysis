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

from sklearn.metrics import accuracy_score, average_precision_score, precision_score,recall_score

from conf import *

import DataReader
import evaluation
#import network
import net as network
import Evaluate
import performance

import cPickle
sys.setrecursionlimit(1000000)

print >> sys.stderr, os.getpid()

if args.language == "en":
    pair_feature_dimention = 70
    mention_feature_dimention = 24
    span_dimention = 5*50
    embedding_dimention = 50
    embedding_size = 34275
    word_embedding_dimention = 8*50
else:
    pair_feature_dimention = 75
    mention_feature_dimention = 23
    span_dimention = 5*64
    embedding_dimention = 64
    embedding_size = 24683
    word_embedding_dimention = 8*64

torch.cuda.set_device(args.gpu)
 
def main():

    DIR = args.DIR
    embedding_file = args.embedding_dir

    #network_file = "./model/model.pkl"
    network_file = "./model/pretrain/network_model_pretrain.49"
    if os.path.isfile(network_file):
        print >> sys.stderr,"Read model from ./model/model.pkl"
        network_model = torch.load(network_file)
    else:
        embedding_matrix = numpy.load(embedding_file)
        #print len(embedding_matrix)

        "Building torch model"
        network_model = network.Network(pair_feature_dimention,mention_feature_dimention,word_embedding_dimention,span_dimention,1000,embedding_size,embedding_dimention,embedding_matrix).cuda()
        print >> sys.stderr,"save model ..."
        torch.save(network_model,network_file)

    reduced=""
    if args.reduced == 1:
        reduced="_reduced"

    print >> sys.stderr,"prepare data for train ..."
    train_docs = DataReader.DataGnerater("train"+reduced)
    print >> sys.stderr,"prepare data for dev and test ..."
    dev_docs = DataReader.DataGnerater("dev"+reduced)
    test_docs = DataReader.DataGnerater("test"+reduced)


    l2_lambda = 1e-6
    lr = 0.00009
    dropout_rate = 0.5
    shuffle = True
    times = 0
    best_thres = 0.5

    model_save_dir = "./model/pretrain/"
   
    last_cost = 0.0
    all_best_results = {
        'thresh': 0.0,
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0
        }
  
    #for echo in range(30,200):
    for echo in range(50,150):

        start_time = timeit.default_timer()
        print "Pretrain Epoch:",echo

        if echo == 100:
            lr = lr*0.7
        #if echo == 150:
        #    lr = lr/2.0

        #optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, network_model.parameters()), lr=lr, weight_decay=l2_lambda)
        #optimizer = optim.RMSprop(network_model.parameters(), lr=lr, weight_decay=l2_lambda)
        optimizer = optim.RMSprop(network_model.parameters(), lr=lr, eps = 1e-5, weight_decay=l2_lambda)

        pair_cost_this_turn = 0.0
        ana_cost_this_turn = 0.0

        pair_nums = 0
        ana_nums = 0

        pos_num = 0
        neg_num = 0
        inside_time = 0.0
        

        for data in train_docs.train_generater(shuffle=shuffle):
            
            mention_word_index, mention_span, candi_word_index,candi_span,feature_pair,pair_antecedents,pair_anaphors,\
            target,positive,negative,anaphoricity_word_indexs, anaphoricity_spans, anaphoricity_features, anaphoricity_target = data
            mention_index = autograd.Variable(torch.from_numpy(mention_word_index).type(torch.cuda.LongTensor))
            mention_span = autograd.Variable(torch.from_numpy(mention_span).type(torch.cuda.FloatTensor))
            candi_index = autograd.Variable(torch.from_numpy(candi_word_index).type(torch.cuda.LongTensor))
            candi_spans = autograd.Variable(torch.from_numpy(candi_span).type(torch.cuda.FloatTensor))
            pair_feature = autograd.Variable(torch.from_numpy(feature_pair).type(torch.cuda.FloatTensor))
            anaphors = autograd.Variable(torch.from_numpy(pair_anaphors).type(torch.cuda.LongTensor))
            antecedents = autograd.Variable(torch.from_numpy(pair_antecedents).type(torch.cuda.LongTensor))

            anaphoricity_index = autograd.Variable(torch.from_numpy(anaphoricity_word_indexs).type(torch.cuda.LongTensor))
            anaphoricity_span = autograd.Variable(torch.from_numpy(anaphoricity_spans).type(torch.cuda.FloatTensor))
            anaphoricity_feature = autograd.Variable(torch.from_numpy(anaphoricity_features).type(torch.cuda.FloatTensor))

            gold = target.tolist()
            anaphoricity_gold = anaphoricity_target.tolist()

            pair_nums += len(gold)
            ana_nums += len(anaphoricity_gold)

            lable = autograd.Variable(torch.cuda.FloatTensor([gold]))
            ana_lable = autograd.Variable(torch.cuda.FloatTensor([anaphoricity_gold]))

            output,_ = network_model.forward_all_pair(word_embedding_dimention,mention_index,mention_span,candi_index,candi_spans,pair_feature,anaphors,antecedents,dropout_rate)
            ana_output,_ = network_model.forward_anaphoricity(word_embedding_dimention, anaphoricity_index, anaphoricity_span, anaphoricity_feature, dropout_rate)

            optimizer.zero_grad()

            #loss = get_pair_loss(output,positive,negative,train_docs.scale_factor)
            loss = F.binary_cross_entropy(output,lable,size_average=False)/train_docs.scale_factor
            ana_loss = F.binary_cross_entropy(ana_output,ana_lable,size_average=False)/train_docs.anaphoricity_scale_factor

            pair_cost_this_turn += loss.data[0]*train_docs.scale_factor
            ana_cost_this_turn += ana_loss.data[0]*train_docs.anaphoricity_scale_factor

            loss_all = loss + ana_loss
            loss_all.backward()
            optimizer.step()

        end_time = timeit.default_timer()
        print >> sys.stderr, "PreTrain epoch",echo,"Pair total cost:",pair_cost_this_turn/float(pair_nums),"Anaphoricity total cost", ana_cost_this_turn/float(ana_nums)
        print >> sys.stderr, "PreTRAINING Use %.3f seconds"%(end_time-start_time)
        print >> sys.stderr, "Learning Rate",lr

        print >> sys.stderr,"save model ..."
        torch.save(network_model, model_save_dir+"network_model_pretrain.%d"%echo)

        #if cost_this_turn > last_cost:
        #    lr = lr*0.7 
        gold = []
        predict = []

        ana_gold = []
        ana_predict = []

        for data in dev_docs.train_generater(shuffle=False):
            
            mention_word_index, mention_span, candi_word_index,candi_span,feature_pair,pair_antecedents,pair_anaphors,\
            target,positive,negative, anaphoricity_word_indexs, anaphoricity_spans, anaphoricity_features, anaphoricity_target = data
         
            mention_index = autograd.Variable(torch.from_numpy(mention_word_index).type(torch.cuda.LongTensor))
            mention_span = autograd.Variable(torch.from_numpy(mention_span).type(torch.cuda.FloatTensor))
            candi_index = autograd.Variable(torch.from_numpy(candi_word_index).type(torch.cuda.LongTensor))
            candi_spans = autograd.Variable(torch.from_numpy(candi_span).type(torch.cuda.FloatTensor))
            pair_feature = autograd.Variable(torch.from_numpy(feature_pair).type(torch.cuda.FloatTensor))
            anaphors = autograd.Variable(torch.from_numpy(pair_anaphors).type(torch.cuda.LongTensor))
            antecedents = autograd.Variable(torch.from_numpy(pair_antecedents).type(torch.cuda.LongTensor))

            anaphoricity_index = autograd.Variable(torch.from_numpy(anaphoricity_word_indexs).type(torch.cuda.LongTensor))
            anaphoricity_span = autograd.Variable(torch.from_numpy(anaphoricity_spans).type(torch.cuda.FloatTensor))
            anaphoricity_feature = autograd.Variable(torch.from_numpy(anaphoricity_features).type(torch.cuda.FloatTensor))

            gold += target.tolist()
            ana_gold += anaphoricity_target.tolist()

            output,_ = network_model.forward_all_pair(word_embedding_dimention,mention_index,mention_span,candi_index,candi_spans,pair_feature,anaphors,antecedents,0.0)
            predict += output.data.cpu().numpy()[0].tolist()

            ana_output,_ = network_model.forward_anaphoricity(word_embedding_dimention, anaphoricity_index, anaphoricity_span, anaphoricity_feature, 0.0)
            ana_predict += ana_output.data.cpu().numpy()[0].tolist()
        
        gold = numpy.array(gold,dtype=numpy.int32)
        predict = numpy.array(predict)

        best_results = {
            'thresh': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }

        thresh_list = [0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]
        for thresh in thresh_list:
            evaluation_results = get_metrics(gold, predict, thresh)
            if evaluation_results["f1"] >= best_results["f1"]:
                best_results = evaluation_results
 
        print "Pair accuracy: %f and Fscore: %f with thresh: %f"\
                %(best_results["accuracy"],best_results["f1"],best_results["thresh"])
        sys.stdout.flush() 

        if best_results["f1"] > all_best_results["f1"]:
            all_best_results = best_results
            print >> sys.stderr, "New High Result, Save Model"
            torch.save(network_model, model_save_dir+"network_model_pretrain.best")

        ana_gold = numpy.array(ana_gold,dtype=numpy.int32)
        ana_predict = numpy.array(ana_predict)
        best_results = {
            'thresh': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
        for thresh in thresh_list:
            evaluation_results = get_metrics(ana_gold, ana_predict, thresh)
            if evaluation_results["f1"] >= best_results["f1"]:
                best_results = evaluation_results
        print "Anaphoricity accuracy: %f and Fscore: %f with thresh: %f"\
                %(best_results["accuracy"],best_results["f1"],best_results["thresh"])
        sys.stdout.flush() 

        if (echo+1)%10 == 0:
            best_network_model = torch.load(model_save_dir+"network_model_pretrain.best") 
            print "DEV:"
            performance.performance(dev_docs,best_network_model)
            print "TEST:"
            performance.performance(test_docs,best_network_model)

    ## output best
    print "In sum, anaphoricity accuracy: %f and Fscore: %f with thresh: %f"\
        %(best_results["accuracy"],best_results["f1"],best_results["thresh"])
    sys.stdout.flush()

def get_metrics(gold, predict, thresh):
    pred = np.clip(np.floor(predict / thresh), 0, 1)
    p, r = (0, 0) if pred.sum() == 0 else \
    (precision_score(gold, pred), recall_score(gold, pred))
    return {
        'thresh': thresh,
        #'accuracy': accuracy_score(gold, pred),
        'accuracy': average_precision_score(gold, predict),
        'precision': p,
        'recall': r,
        'f1': 0 if p == 0 or r == 0 else 2 * p * r / (p + r)
    } 

def get_pair_loss(output,pos_index,neg_index,scale_factor):

    pos = torch.from_numpy(pos_index).type(torch.cuda.LongTensor)
    neg = torch.from_numpy(neg_index).type(torch.cuda.LongTensor)
    #return -(torch.sum(torch.log(output[0][pos] + 1e-9))+torch.sum(torch.log(1-output[0][neg]+ 1e-9)))/scale_factor
    return -(torch.mean(torch.log(output[0][pos] + 1e-9))+torch.mean(torch.log(1-output[0][neg]+ 1e-9)))/scale_factor
    #return -(torch.mean(torch.log(output[0][pos] + 1e-12))+torch.mean(torch.log(1-output[0][neg]+ 1e-12)))


if __name__ == "__main__":
    main()
