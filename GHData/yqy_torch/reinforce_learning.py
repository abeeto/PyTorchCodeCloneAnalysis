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
import utils
import performance

import cPickle
sys.setrecursionlimit(1000000)

print >> sys.stderr, os.getpid()

if args.language == "en":
    pair_feature_dimention = 77
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
    #network_file = "./model/pretrain/network_model_pretrain.20"
    network_file = "./model/pretrain/network_model_pretrain.top.best"
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
    #train_docs = DataReader.DataGnerater("dev"+reduced)
    print >> sys.stderr,"prepare data for dev and test ..."
    dev_docs = DataReader.DataGnerater("dev"+reduced)
    #test_docs = DataReader.DataGnerater("test"+reduced)


    l2_lambda = 1e-6
    lr = 0.00002
    dropout_rate = 0.5
    shuffle = True
    times = 0
    best_thres = 0.5

    reinforce = True

    model_save_dir = "./model/pretrain/"
   
    metrics = performance.performance(dev_docs,network_model) 

    p,r,f = metrics["b3"]

    f_b = [f]
  
    #for echo in range(30,200):
    for echo in range(20):

        start_time = timeit.default_timer()
        print "Pretrain Epoch:",echo

        #if echo == 100:
        #    lr = lr/2.0
        #if echo == 150:
        #    lr = lr/2.0

        #optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, network_model.parameters()), lr=lr, weight_decay=l2_lambda)
        #optimizer = optim.RMSprop(network_model.parameters(), lr=lr, weight_decay=l2_lambda)
        cost = 0.0
        optimizer = optim.RMSprop(network_model.parameters(), lr=lr, eps = 1e-5, weight_decay=l2_lambda)

        pair_cost_this_turn = 0.0
        ana_cost_this_turn = 0.0

        pair_nums = 0
        ana_nums = 0

        pos_num = 0
        neg_num = 0
        inside_time = 0.0
        
        score_softmax = nn.Softmax()
        
        cluster_info = []
        new_cluster_num = 0 
        cluster_info.append(-1)
        action_list = []
        new_cluster_info = []
        tmp_data = []

        #for data in train_docs.rl_case_generater():
        for data in train_docs.rl_case_generater(shuffle=True):
            inside_time += 1
            
            this_doc = train_docs
            tmp_data.append(data)
            
            mention_word_index, mention_span, candi_word_index,candi_span,feature_pair,pair_antecedents,pair_anaphors,\
            target,positive,negative,anaphoricity_word_indexs, anaphoricity_spans, anaphoricity_features, anaphoricity_target,rl,candi_ids_return = data

            gold_chain = this_doc.gold_chain[rl["did"]]
            gold_dict = {}
            for chain in gold_chain:
                for item in chain:
                    gold_dict[item] = chain

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

            output, pair_score = network_model.forward_all_pair(word_embedding_dimention,mention_index,mention_span,candi_index,candi_spans,pair_feature,anaphors,antecedents,dropout_rate)
            ana_output, ana_score = network_model.forward_anaphoricity(word_embedding_dimention, anaphoricity_index, anaphoricity_span, anaphoricity_feature, dropout_rate)

            reindex = autograd.Variable(torch.from_numpy(rl["reindex"]).type(torch.cuda.LongTensor))

            scores_reindex = torch.transpose(torch.cat((pair_score,ana_score),1),0,1)[reindex]
            #scores_reindex = torch.transpose(torch.cat((pair_score,-1-0.3*ana_score),1),0,1)[reindex]

            for s,e in zip(rl["starts"],rl["ends"]):
                #action_prob: scores_reindex[s:e][1]
                score = score_softmax(torch.transpose(scores_reindex[s:e],0,1)).data.cpu().numpy()[0]
                this_action = utils.sample_action(score)
                #this_action = ac_list.index(max(score.tolist())) 
                action_list.append(this_action)
                
                if this_action == len(score)-1 :
                    should_cluster = new_cluster_num
                    new_cluster_num += 1
                    new_cluster_info.append(1)
                else:
                    should_cluster = cluster_info[this_action]
                    new_cluster_info.append(0)

                cluster_info.append(should_cluster)

            if rl["end"] == True:
                ev_document = utils.get_evaluation_document(cluster_info,this_doc.gold_chain[rl["did"]],candi_ids_return,new_cluster_num)
                p,r,f = evaluation.evaluate_documents([ev_document],evaluation.b_cubed)
                trick_reward = utils.get_reward_trick(cluster_info,gold_dict,new_cluster_info,action_list,candi_ids_return)

                #reward = f + trick_reward
                average_f = float(sum(f_b))/len(f_b)

                reward = (f - average_f)*10

                f_b.append(f)
                if len(f_b) > 128:
                    f_b = f_b[1:]

                index = 0
                for data in tmp_data:
                    mention_word_index, mention_span, candi_word_index,candi_span,feature_pair,pair_antecedents,pair_anaphors,\
                    target,positive,negative,anaphoricity_word_indexs, anaphoricity_spans, anaphoricity_features, anaphoricity_target,rl,candi_ids_return = data

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

                    rl_costs = autograd.Variable(torch.from_numpy(rl["costs"]).type(torch.cuda.FloatTensor))
                    rl_costs = torch.transpose(rl_costs,0,1)

                    output, pair_score = network_model.forward_all_pair(word_embedding_dimention,mention_index,mention_span,candi_index,candi_spans,pair_feature,anaphors,antecedents,dropout_rate)
                    ana_output, ana_score = network_model.forward_anaphoricity(word_embedding_dimention, anaphoricity_index, anaphoricity_span, anaphoricity_feature, dropout_rate)

                    reindex = autograd.Variable(torch.from_numpy(rl["reindex"]).type(torch.cuda.LongTensor))

                    optimizer.zero_grad()
                    loss = None
                    scores_reindex = torch.transpose(torch.cat((pair_score,ana_score),1),0,1)[reindex]
                    #scores_reindex = torch.transpose(torch.cat((pair_score,-1-0.3*ana_score),1),0,1)[reindex]
                    
                    for s,e in zip(rl["starts"],rl["ends"]):
                        #action_prob: scores_reindex[s:e][1]
                        this_action = action_list[index]
                        #current_reward = reward + trick_reward[index]
                        current_reward = reward

                        #this_loss = -reward*(torch.transpose(F.log_softmax(torch.transpose(scores_reindex[s:e],0,1)),0,1)[this_action])
                        this_loss = -current_reward*(torch.transpose(F.log_softmax(torch.transpose(scores_reindex[s:e],0,1)),0,1)[this_action])

                        if loss is None:
                            loss = this_loss
                        else:
                            loss += this_loss
                        index += 1 
                    #loss /= len(rl["starts"])
                    loss /= len(rl["starts"])
                    #loss = loss/train_docs.scale_factor
                    ## policy graident
                    cost += loss.data[0]
                    loss.backward()
                    optimizer.step()

                new_cluster_num = 0
                cluster_info = []
                cluster_info.append(-1)
                tmp_data = []
                action_list = []
                new_cluster_info = []
            #if inside_time%50 == 0:
            #    performance.performance(dev_docs,network_model) 
            #    print 
            #    sys.stdout.flush()
     
        end_time = timeit.default_timer()
        print >> sys.stderr, "PreTRAINING Use %.3f seconds"%(end_time-start_time)
        print >> sys.stderr, "cost:",cost
        #print >> sys.stderr,"save model ..."
        #torch.save(network_model, model_save_dir+"network_model_pretrain.%d"%echo)
        
        performance.performance(dev_docs,network_model) 

        sys.stdout.flush()

if __name__ == "__main__":
    main()
