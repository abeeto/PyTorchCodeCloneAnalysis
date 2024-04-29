#coding=utf8

import sys
import os
import json
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
import net as network
import utils

import cPickle
sys.setrecursionlimit(1000000)

if args.language == "en":
    word_embedding_dimention = 8*50
else:
    word_embedding_dimention = 8*64

torch.cuda.set_device(args.gpu)

def evaluate(network_model,doc,best_thres=0.5):

    best_thres = best_thres

    docs_for_test = []
    gold_docs_for_test = []
    all_docs_for_test = []

    gold_anaphor_for_test = []
    predict_anaphor_for_test = []
        
    action_list = []
    ana_probs = []
    coref_probs = []
    gold_anaphora = []
    last_candidates_id = []

    for data,doc_end in doc.generater():
        ana_word_index,ana_span,ana_feature,candi_word_index,candi_span,pair_feature_array,target,mention_ids = data
        did,mention_id,pair_candidates_id = mention_ids 

        ## generate actions ...

        if len(target) == 0:

            ana_probs.append(0.0)
            coref_probs.append([])
            this_action = -1 

        else:

            mention_index = autograd.Variable(torch.from_numpy(ana_word_index).type(torch.cuda.LongTensor))
            mention_span = autograd.Variable(torch.from_numpy(ana_span).type(torch.cuda.FloatTensor))
            mention_feature = autograd.Variable(torch.from_numpy(ana_feature).type(torch.cuda.FloatTensor))
            candi_index = autograd.Variable(torch.from_numpy(candi_word_index).type(torch.cuda.LongTensor))
            candi_spans = autograd.Variable(torch.from_numpy(candi_span).type(torch.cuda.FloatTensor))
            pair_feature = autograd.Variable(torch.from_numpy(pair_feature_array).type(torch.cuda.FloatTensor))

            output,_ = network_model.forward(word_embedding_dimention,mention_index,mention_span,mention_feature,mention_index,mention_span,candi_index,candi_spans,pair_feature,0.0)
            output = output.data.cpu().numpy()[0]

            zero_score = output[0]
            coref_prob = output[1:]

            ana_probs.append(1-zero_score)
            coref_probs.append(coref_prob)
                
            ac_list = list(coref_prob)
            this_action = ac_list.index(max(ac_list))

        action_list.append(this_action)
        last_candidates_id.append(mention_id)

        if sum(target) == 0:
            gold_anaphora.append(0)
        else:
            gold_anaphora.append(1)

        if doc_end:
            ## evaluation
            gold_chain = doc.gold_chain[did]
                
            # evaluate anaphora result
            gold_anaphor_for_test += gold_anaphora
            predict_anaphor_for_test += ana_probs

            cluster_info = []
            new_cluster_num = 0 
            for action,ana_prob,ana_gold,coref_prob in zip(action_list,ana_probs,gold_anaphora,coref_probs):
                ## gold results: 
                if  ana_gold == 0: # it is not an anphoric mention
                    should_cluster = new_cluster_num
                    new_cluster_num += 1
                else:
                    should_cluster = cluster_info[action]
                cluster_info.append(should_cluster)
            ev_document = utils.get_evaluation_document(cluster_info,gold_chain,last_candidates_id,new_cluster_num)
            gold_docs_for_test.append(ev_document)

            cluster_info = []
            new_cluster_num = 0 
            for action,ana_prob,ana_gold,coref_prob in zip(action_list,ana_probs,gold_anaphora,coref_probs):
                ## predict results: 
                if  ana_prob < best_thres: # it is not an anphoric mention
                    should_cluster = new_cluster_num
                    new_cluster_num += 1
                else:
                    #print action,cluster_info
                    should_cluster = cluster_info[action]
                
                cluster_info.append(should_cluster)
            ev_document = utils.get_evaluation_document(cluster_info,gold_chain,last_candidates_id,new_cluster_num)
            docs_for_test.append(ev_document)

            cluster_info = []
            new_cluster_num = 0 
            for action,ana_prob,ana_gold,coref_prob in zip(action_list,ana_probs,gold_anaphora,coref_probs):
            ## predict results with all things: 

                ac_list = [1-ana_prob] + list(coref_prob)
                this_action = ac_list.index(max(ac_list))

                if  this_action == 0: # it is not an anphoric mention
                    should_cluster = new_cluster_num
                    new_cluster_num += 1
                else:
                    should_cluster = cluster_info[this_action-1]

                cluster_info.append(should_cluster)
            ev_document = utils.get_evaluation_document(cluster_info,gold_chain,last_candidates_id,new_cluster_num)
            all_docs_for_test.append(ev_document)

            action_list = []
            ana_probs = []
            coref_probs = []
            gold_anaphora = []
            last_candidates_id = []

  
    p,r,f,thred = utils.get_prf_probs(gold_anaphor_for_test,predict_anaphor_for_test) 
    print "for Anaphority Identification"
    print "P: %f R: %f F: %f with thred %f"%(p,r,f,thred)
    best_thres = thred

    print "Test on ALL classfication"  
    m = Output_Result(all_docs_for_test)
    print_performance(m)
    print "Test on GOLD anaphor"  
    m = Output_Result(gold_docs_for_test)
    print_performance(m)
    print "Test on PREDICT anaphor"  
    m = Output_Result(docs_for_test)
    print_performance(m)

    print "#################################################" 
    sys.stdout.flush()

    return best_thres    

def Output_Result(doc4test):
    mp,mr,mf = evaluation.evaluate_documents(doc4test,evaluation.muc)
    #print "MUC: recall: %f precision: %f  f1: %f"%(mr,mp,mf)
    bp,br,bf = evaluation.evaluate_documents(doc4test,evaluation.b_cubed)
    #print "BCUBED: recall: %f precision: %f  f1: %f"%(br,bp,bf)
    cp,cr,cf = evaluation.evaluate_documents(doc4test,evaluation.ceafe)
    #print "CEAF: recall: %f precision: %f  f1: %f"%(cr,cp,cf)
    metrics = {}
    metrics["muc"] = (mr,mp,mf)
    metrics["b3"] = (br,bp,bf)
    metrics["ceaf"] = (cr,cp,cf)
    return metrics
def print_performance(m):
    mp,mr,mf = m["muc"]
    print "MUC: recall: %f precision: %f  f1: %f"%(mr,mp,mf)
    bp,br,bf = m["b3"]
    print "BCUBED: recall: %f precision: %f  f1: %f"%(br,bp,bf)
    cp,cr,cf = m["ceaf"]
    print "CEAF: recall: %f precision: %f  f1: %f"%(cr,cp,cf)
     
if __name__ == "__main__":

    #network_file = "./model/pretrain/network_model_pretrain.best"
    network_file = "./model/pretrain/network_model_pretrain.top.best"
    #network_file = "./model/model.pkl"
    print >> sys.stderr,"Read model from ./model/model.pkl"
    network_model = torch.load(network_file)

    #dev_docs = DataReader.DataGnerater("dev")
    dev_docs = DataReader.DataGnerater("test")

    best_thres = 0.4

    best_thres = evaluate(network_model,dev_docs,best_thres)

