#coding=utf8

import sys
import os
import json
import random
import numpy
import timeit

from conf import *

import evaluation
from network import *

import cPickle
sys.setrecursionlimit(1000000)

random.seed(args.random_seed)

def sample_action(action_probability):
    
    #ac = list(action_probability)
    #ac = numpy.array(ac)
    ac = action_probability
    ac = ac/ac.sum()
    action = numpy.random.choice(numpy.arange(len(ac)),p=ac)
    return action

def sample_action_trick(action_probability,ran_p = 0.05):
    if random.random() <= ran_p:
        action = random.randint(0,len(action_probability)-1)
        return action
    else:
        ac = action_probability/action_probability.sum()
        action = numpy.random.choice(numpy.arange(len(ac)),p=ac)
        return action


def choose_action(action_probability):
    #print action_probability
    ac_list = list(action_probability)
    action = ac_list.index(max(ac_list))
    return action

def get_reward(cluster_info,gold_info,max_cluster_num):
    ev_document = get_evaluation_document(cluster_info,gold_info,max_cluster_num)
    p,r,f = evaluation.evaluate_documents([ev_document],evaluation.b_cubed)
    #return f
    return p,r,f

def get_reward_average(cluster_info,gold_info,max_cluster_num,index,max_cluster_index):
    # build new cluster
    new_cluster_prefix = cluster_info[:index]
    new_cluster_postfix = cluster_info[index+1:]

    el = []
    
    for cluster_num in range(max_cluster_index):
        new_cluster_info = new_cluster_prefix + [cluster_num] + new_cluster_postfix 
        ev_document = get_evaluation_document(new_cluster_info,gold_info,max_cluster_num)
        el.append(ev_document)
    p,r,f = evaluation.evaluate_documents(el,evaluation.b_cubed)
    #p,r,f = evaluation.evaluate_documents([ev_document],evaluation.muc)
    #print >> sys.stderr, p,r,f
    return f


def get_reward_trick(cluster_info,gold_dict,new_cluster_info,action_list,candi_ids):

    award_list = []

    for i in range(1,len(cluster_info)):
        this_cluster = cluster_info[i]
        this_candi_id = candi_ids[i]
        this_action = action_list[i-1]
        new_cluster = new_cluster_info[i-1]

        if this_candi_id in gold_dict: # should not be a new one
            if new_cluster == 1:
                this_award = -1
            else:
                if candi_ids[this_action] in gold_dict[this_candi_id]:
                    this_award = 1
                else:
                    this_award = -1
        else: # should be a new
            if new_cluster == 1:
                this_award = 1
            else:
                this_award = -1
        award_list.append(this_award)

    return award_list
         

def get_evaluation_document(cluster_info,gold_info,doc_ids,max_cluster_num):
    predict = []
    predict_dict = {}
   
    for mention_num in range(len(cluster_info)):
        cluster_num = cluster_info[mention_num]
        predict_dict.setdefault(cluster_num,[])
        predict_dict[cluster_num].append(doc_ids[mention_num])
        #predict[cluster_num].append(mention_num)
    for k in sorted(predict_dict.keys()):
        predict.append(predict_dict[k])
    ev_document = evaluation.EvaluationDocument(gold_info,predict)
    return ev_document

def get_prf_probs(gold,probs = None):
    if probs is not None:
        tuning_probs = [0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
        best_f = 0.0
        best_p = 0.0
        best_r = 0.0
        best_prob = 0.0
        for prob_thres in tuning_probs:
            new_predict = []
            for prob in probs:
                if prob > prob_thres: # if is equls to 1 then is anaphoric
                    new_predict.append(1) # is anaphoric
                else:
                    new_predict.append(0)
        
            pr = 0
            should = 0
            r_in_r = 0
            for i in range(len(new_predict)):
                if new_predict[i] == 1: ## is anaphoric == postive example
                    pr += 1 
                    if gold[i] == 1:
                        r_in_r += 1
                if gold[i] == 1:
                    should += 1
            if should == 0 or pr == 0:
                continue
            else:
                p = float(r_in_r)/float(pr)
                r = float(r_in_r)/float(should)
                if (not p == 0) and (not r == 0):
                    f = 2.0/(1.0/r+1.0/p)
                else:
                    f = 0.0
                if f >= best_f:
                    best_f = f
                    best_p = p
                    best_r = r
                    best_prob = prob_thres
        print "Got best in ",best_prob
        return best_p,best_r,best_f,best_prob


def get_prf(predict,gold):
    if predict is not None:
        pr = 0
        should = 0
        r_in_r = 0
        for i in range(len(predict)):
            if not predict[i] == -1: ## a new cluster
                pr += 1 
                if gold[i] == 1:
                    r_in_r += 1
            if gold[i] == 1:
                should += 1
        if should == 0:
            return 0.0,0.0,0.0
        elif pr == 0:
            return 0.0,0.0,0.0
        else:
            p = float(r_in_r)/float(pr)
            r = float(r_in_r)/float(should)
            if (not p == 0) and (not r == 0):
                f = 2.0/(1.0/r+1.0/p)
            else:
                f = 0.0
            return p,r,f
