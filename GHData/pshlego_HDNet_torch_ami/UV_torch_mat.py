"""
author: Yasamin Jafarian
"""

#import tensorflow as tf
from ast import Str
from os import path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import skimage.data
from PIL import Image, ImageDraw, ImageFont
import random
import sys
import matplotlib.pyplot as plt
#tf.logging.set_verbosity(tf.logging.ERROR)
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pdb
import math
#from tensorflow.python.platform import gfile
import scipy.misc
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import time
from hourglass_net_depth_torch import hourglass_refinement_1
from hourglass_net_normal_torch import hourglass_normal_prediction_1
from utils import (get_origin_scaling,get_concat_h, depth2mesh, read_test_data, nmap_normalization, get_test_data,get_tiktok_data,read_tiktok_data) 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
start_time_1 = time.time()
############################## test path and outpath ##################################
img_number="/00138/"
#data_main_path = '/local_data/TikTok_dataset'+img_number
data_main_path = './training_data/tiktok_data'
#data_main_path = './test_data'+img_number
#outpath = './test_data'+"/infer_out/"+"pytorch/"+"HDNet_up6"+img_number
outpath_1 = './UV_result/corrs/'
outpath = './UV_result/'
visualization = True
num=306
##############################    Inference Code     ##################################
#pre_ck_pnts_dir_DR =  "/home/ug_psh/HDNet_torch_ami/training_progress/pytorch/model/HDNet/"
pre_ck_pnts_dir_DR =  "/home/ug_psh/HDNet_torch_ami/training_progress/pytorch/model/HDNet_up6/"
model_num_DR = '1920000'
#pre_ck_pnts_dir_NP =  "/home/ug_psh/HDNet_torch_ami/training_progress/pytorch/model/NormalEstimator/"
pre_ck_pnts_dir_NP =  "/home/ug_psh/HDNet_torch_ami/training_progress/pytorch/model/HDNet_up6/"
model_num_NP = '1710000'
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
def write_matrix_txt(a,filename):
    mat = np.matrix(a)
    with open(filename,'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.1f')
# Creat the outpath if not exists
Vis_dir = outpath
if not path.exists(Vis_dir):
    print("Vis_dir created!")
    os.makedirs(Vis_dir)
#refineNet_graph = tf.Graph()
#NormNet_graph = tf.Graph()


# Read the test images and run the HDNet
test_files = get_tiktok_data(data_main_path,num)
X_1 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3,num),dtype='f')
for f in range(len(test_files)):
    data_name = test_files[f]
    #print('Processing file: ',data_name)
    DP = read_tiktok_data(data_main_path,data_name,IMAGE_HEIGHT,IMAGE_WIDTH)
    X_1[...,f]=DP
    
    #pdb.set_trace()
    #write_matrix_txt(DP[...,0],Vis_dir+data_name+"_V.txt")
    #write_matrix_txt(DP[...,1],Vis_dir+data_name+"_U.txt")
    #write_matrix_txt(DP[...,2],Vis_dir+data_name+"_I.txt")
#generate corr_mat.txt
frame_list=torch.zeros((num,num)).type(torch.int).to(device)
frame_index=torch.zeros((num,2,num)).type(torch.int).to(device)
#pdb.set_trace()
#X_wo_z=np.setdiff1d(np.reshape(X_1[0,:,:,:,0],(256*256,3))[:,2],np.zeros((256*256,1)))
# X_wo_z=np.concatenate((np.reshape(X_1[0,:,:,:,0],(256*256,3)),np.reshape(X_1[0,:,:,:,0],(256*256,3))),axis=0)
# X_count=np.unique(X_wo_z, return_counts=True, axis=0)
# X_final=np.where(X_count[1]>=2, X_count[0][:,2],0)
# count=0
# for i in range(24):
#     if np.sum(X_final==i+1)>=50:
#         count+=1
corr_mat_tik = np.genfromtxt(Vis_dir +'corr_mat.txt',delimiter=",")
#pdb.set_trace()
for g in range(np.shape(corr_mat_tik)[0]):
    for v in range(np.shape(corr_mat_tik)[1]-1):
        k=0
        corres=[]
        corres_body=[]
        dist_1=[]
        pdist = nn.PairwiseDistance(p=2)
        x_2=torch.Tensor(X_1[0,:,:,:,int(corr_mat_tik[g][0])-17633]).type(torch.float).to(device)#0017634
        x_2_1=torch.Tensor(X_1[0,:,:,:,int(corr_mat_tik[g][v+1])-17633]).type(torch.float).to(device)#0017637
        #pdb.set_trace()
        #x_2[...,0][x_2[...,2]==2] 0017634에서 body part 2인 V torch.Size([6113])
        #x_2[...,0][x_2[...,2]==2] 0017634에서 body part 2인 U torch.Size([6113])
        bpbp=np.array(range(26))
        for j in range(24):
            body_part=0
            index=torch.where(x_2[...,2]==j+1)
            indicies=torch.stack([index[0],index[1]],dim=1)
            size=int(x_2[...,0][x_2[...,2]==j+1].size()[0])
            x_2_vu=torch.cat((x_2[...,0][x_2[...,2]==j+1].view(size,1),x_2[...,1][x_2[...,2]==j+1].view(size,1)),dim=1)
            record=torch.zeros_like(x_2_vu).type(torch.float).to(device)
            #x_2_1[...,0][x_2_1[...,2]==2] 0017637에서 body part 2인 V torch.Size([6702])
            #x_2_1[...,0][x_2_1[...,2]==2] 0017637에서 body part 2인 U torch.Size([6702])
            index_1=torch.where(x_2_1[...,2]==j+1)
            indicies_1=torch.stack([index_1[0],index_1[1]],dim=1)
            size_1=int(x_2_1[...,0][x_2_1[...,2]==j+1].size()[0])
            x_2_1_vu=torch.cat((x_2_1[...,0][x_2_1[...,2]==j+1].view(size_1,1),x_2_1[...,1][x_2_1[...,2]==j+1].view(size_1,1)),dim=1)
            for i in range(size):
                
                xx=torch.repeat_interleave(x_2_vu[i,:].view(1,2),size_1, dim=0)
                record[i,:]=x_2_vu[i,:]
                dupl=torch.sum(torch.unique(record, return_counts=True, dim=0)[1]>=2)==1
                #pdb.set_trace()
                if dupl==False:
                    record[i,:]=torch.tensor([0,0]).type(torch.float).to(device)
                    #pdb.set_trace()    
                else:
                    dist=pdist(xx,x_2_1_vu)
                    if np.shape(dist)[0]>=1:
                        aa=torch.topk(dist, 1, largest=False)
                        #pdb.set_trace()
                    else:
                        aa=[[1000]]
                    if aa[0][0]<7:
                        corres.append([])
                        dist_1.append([])
                        res=np.concatenate((np.array([j+1]),np.array(indicies[i,:].detach().cpu().numpy())),axis=0)
                        res=np.concatenate((res,np.array(indicies_1[aa[1][0],:].detach().cpu().numpy())),axis=0)
                        corres[k].append(res)
                        dist_1[k].append(np.array([aa[0][0].detach().cpu().numpy()]))
                        k+=1
                        body_part+=1
            bpbp[j+1]=k
            if body_part==0:
                corres_body.append([])
                res=np.concatenate((np.array([j+1]),np.array([-1,-1])),axis=0)
                corres_body[j].append(res)
            else:
                corres_body.append([])
                res=np.concatenate((np.array([j+1]),np.array([bpbp[j],bpbp[j+1]-1])),axis=0)
                corres_body[j].append(res)

        np.savetxt(outpath_1+'00'+str(int(corr_mat_tik[g][0]))+'_'+'00'+str(int(corr_mat_tik[g][v+1]))+'_i_r1_c1_r2_c2.txt', np.reshape(np.array(corres),(-1,5)), delimiter=",", fmt="%s")
        #pdb.set_trace()
        #np.savetxt(Vis_dir+'/corrs/'+int(corr_mat_tik[g][0])+'_'+int(corr_mat_tik[g][v+1])+'_'+'dist.txt', np.reshape(np.array(dist_1),(k,-1)), delimiter=",", fmt="%s")
        np.savetxt(outpath_1+'00'+str(int(corr_mat_tik[g][0]))+'_'+'00'+str(int(corr_mat_tik[g][v+1]))+'_i_limit.txt', np.reshape(np.array(corres_body),(24,-1)), delimiter=",", fmt="%s")
        print('00'+str(int(corr_mat_tik[g][0]))+'_'+'00'+str(int(corr_mat_tik[g][v+1])))