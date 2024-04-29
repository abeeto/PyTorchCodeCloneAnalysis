"""
Created on June 24 2019

@author: Finterly
"""
import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torch.autograd import Variable   
import numpy as np
import pandas as pd
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import csv
import itertools

def calc_statistics(my_file, db, tool):    
    target=[]
    prediction=[]
    with open(my_file,'r') as f:
    #     next(f) # skip headings
        reader=csv.reader(f,delimiter='\t')
        for true,pred in reader:
            true = int(true)
            pred = float(pred)
            target.append(true)
            prediction.append(pred) 
    print(db, tool, auc_scores(prediction, target))
        
def auc_scores(y_scores, y_true):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores) 
    auc = metrics.auc(fpr, tpr) 
    auprc = metrics.average_precision_score(y_true, y_scores) 
    auc1000 = aucNth(y_true, y_scores, 1000)
    return auc, auprc, auc1000

def aucNth(y_true, y_pred, N):
    assert len(y_true) == len(y_pred)
    assert len(y_true) > 1
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    negatives = y_true.count(0)
    assert N < negatives
    perc = N / float(negatives)    
    fpr1k = []
    tpr1k = []
    i = 0
    while i < len(fpr):        
        if fpr[i] > perc:
            break
        fpr1k.append(fpr[i])
        tpr1k.append(tpr[i])
        i+=1    
    assert len(fpr1k) > 1
    #print fpr1k, tpr1k
    aucScore = metrics.auc(fpr1k, tpr1k) / perc     
    return aucScore    

db_list = ['pfam', 'gene3d', 'supfam']
tool_list = ['csblast','phmmer','hhsearch','blast','usearch','fasta','ublast']
arg_list = list(itertools.product(db_list,tool_list))
for args in arg_list:
    db = args[0]
    tool = args[1]
    evalue_file =  os.path.join(os.path.expanduser('~'), 'thesis', 'saripella_repository', 'Bitscores_and_Evalues', 'Evalues_max50_1vs1_label', tool + '_' + db + '_max50_1vs1' )
    calc_statistics(evalue_file, db, tool)
# 
# db = "supfam"
# layer = 4
# my_dir= os.path.join(os.path.expanduser("~"), "thesis", "thesis_eclipse", "data","results", "jaccard_analysis" )
# my_file = os.path.join(my_dir, db+ '_' + str(layer) + '_layer')
# calc_statistics(my_file, db, str(layer))

