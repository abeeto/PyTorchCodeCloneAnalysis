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
import csv

def calc_statistics(my_file):    
    target=[]
    prediction=[]
    with open(my_file,'r') as f:
        reader=csv.reader(f,delimiter='\t')
        for true,pred in reader:
            true = int(true)
            pred = float(pred)
            target.append(true)
            prediction.append(pred) 
    return target, prediction 

def plot_roc_curve(fpr, tpr, roc_auc, tool, color):
    lw = 6
    plt.plot(fpr, tpr, color=color, lw = lw, label = tool + ' AUC = ' + str(round(roc_auc, 2)))
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(db.upper() + ' ROC Curve')
    plt.legend(loc="lower right")
    
def plot_pr_curve(recall, precision, auprc, tool, color):
    lw = 6
    plt.plot(recall, precision, color=color, lw = lw, label = tool + ' AUPRC = ' + str(round(auprc, 2)))
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(db.upper() + ' Precision-Recall Curve')
    plt.legend(loc="lower right")    

from pycm import *
  
def plot_now(my_file, tool, color):
    target, prediction =   calc_statistics(my_file)   
    fpr, tpr, thresholds = metrics.roc_curve(target, prediction)
    roc_auc = metrics.auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, tool, color)  
    
def plot_now2(my_file, tool, color):
    target, prediction =   calc_statistics(my_file)   
    precision, recall, thresholds = metrics.precision_recall_curve(target, prediction)
    auprc = metrics.average_precision_score(target, prediction)
    plot_pr_curve(recall, precision, auprc, tool, color)       

     
db = "supfam"
my_dir= os.path.join(os.path.expanduser("~"), "thesis", "thesis_eclipse", "data","results", "jaccard_analysis" )
evalue_dir = os.path.join(os.path.expanduser("~"), "thesis", "thesis_eclipse", "data","results", "jaccard_analysis" , "Evalues_max50")
tool_list = ['csblast','phmmer','hhsearch','blast','usearch','fasta','ublast']

plt.rcParams.update({'font.size': 30})
plt.rcParams["font.family"] = "serif"
plt.figure(0).clf()
plt.figure(figsize=(15,12))
my_file = os.path.join(my_dir, db+ '_4_layer')
plot_now(my_file,'SiameseNN', "#009E73")
color_list = ["#E69F00", "#56B4E9", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#E695B8"]
i = 0
for tool in tool_list: 
    toolpath = os.path.join(evalue_dir, tool +"_" + db + "_max50_1vs1")
    plot_now(toolpath, tool.upper(), color_list[i])
    i = i + 1
plt.legend(loc=0)
plt.savefig(os.path.join(my_dir, db+'_roc_curve.pdf'))

plt.figure(0).clf()
plt.figure(figsize=(15,12))
plot_now2(my_file,'SiameseNN', "#009E73")
color_list = ["#E69F00", "#56B4E9", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#E695B8"]
i = 0
for tool in tool_list: 
    toolpath = os.path.join(evalue_dir, tool +"_" + db + "_max50_1vs1")
    plot_now2(toolpath, tool.upper(), color_list[i])
    i = i + 1
plt.legend(loc=0)
plt.savefig(os.path.join(my_dir, db+'_pr_curve.pdf'))




