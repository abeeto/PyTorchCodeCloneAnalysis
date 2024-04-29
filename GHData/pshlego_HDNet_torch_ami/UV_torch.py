"""
author: Yasamin Jafarian
"""

#import tensorflow as tf
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
k=0
x_2=torch.Tensor(X_1[0,:,:,:,1]).type(torch.int).to(device)
cc=0
#for i in range(256):
#    for j in range(256):
#        if x_2[i][j][0]==78 and x_2[i][j][1]==71 and x_2[i][j][2]==2:
#            cc+=1
#pdb.set_trace()
corr_mat_let=[]
c_f_let=[]
for f in range(int(num/3)):
    start_time = time.time()
    c_f=0
    x_2=torch.Tensor(X_1[0,:,:,:,3*f+1]).type(torch.int).to(device)
    for j in range(int(num/3)):
        x_2_1=torch.Tensor(X_1[0,:,:,:,3*j+1]).type(torch.int).to(device)
        X_wo_z_1=x_2.view(256*256,3)
        X_wo_z_2=x_2_1.view(256*256,3)
        X_count_1=torch.unique(X_wo_z_1, return_counts=True, dim=0)
        X_count_2=torch.unique(X_wo_z_2, return_counts=True, dim=0)
        
        X_wo_z=torch.cat((X_count_1[0],X_count_2[0]),dim=0)
        #X_wo_z=x_2.view(256*256,3)
        X_count=torch.unique(X_wo_z, return_counts=True, dim=0)
        X_final=torch.where(X_count[1]==2, X_count[0][:,2], torch.zeros_like(X_count[0][:,2]).type(torch.int).to(device)).type(torch.int).to(device)
        #pdb.set_trace()
        count=0
        #np.savetxt(Vis_dir+'corr_0.txt', X_wo_z.detach().cpu().numpy(), delimiter=",", fmt="%s")
        #np.savetxt(Vis_dir+'corr_1.txt', X_count[0].detach().cpu().numpy(), delimiter=",", fmt="%s")
        #np.savetxt(Vis_dir+'corr_2.txt', X_count[1].detach().cpu().numpy(), delimiter=",", fmt="%s")
        #pdb.set_trace()
        count=torch.sum(torch.unique(torch.sort(X_final)[0],return_counts=True)[1]>=30)
        # for i in range(24):
        #     if torch.sum(X_final==i+1)>=50:
        #         count+=1
        #pdb.set_trace()
        # if count>=5:
        #     frame_list[f,j]=1
        #print(count)
        if count>=6:
            frame_list[3*f+1,3*j+1]=1
            frame_index[c_f,:,3*f+1]=torch.Tensor([3*f+1,3*j+1]).type(torch.int).to(device)
            c_f+=1
            #pdb.set_trace()
    
    print("1 frame Time taken: %.2fs %.2f" % (time.time() - start_time,c_f))
    #pdb.set_trace()
    #if (c_f-1)>=49:
        #print(c_f)
    # if c_f-1>=5:
    
    #     corr_mat_let.append([])
    #     rd=random.sample(range(0, c_f-1), 5)
    #     index=np.sort(torch.gather(frame_index[:,1,3*f+1],0,torch.Tensor(np.array(rd)).type(torch.long).to(device)).detach().cpu().numpy())
    #     index[0]=3*f+1
    #     corr_mat_let[cc].append(np.reshape(np.array([np.take(np.array(test_files),index)[0][2:7],np.take(np.array(test_files),index)[1][2:7],np.take(np.array(test_files),index)[2][2:7],np.take(np.array(test_files),index)[3][2:7],np.take(np.array(test_files),index)[4][2:7]]),(5,)))
    #     #k+=1
    #     cc+=1
    #pdb.set_trace()
    frame_index[c_f:c_f*2-1,:,3*f+1]=torch.from_numpy(np.flip(frame_index[0:c_f-1,:,3*f+1].detach().cpu().numpy(),axis=0).copy()).to(device)
    if c_f-1>=5:
        corr_mat_let.append([])
        c_f_let.append([])
        lf=torch.arange(c_f)
        ind=lf[(frame_index[:,1,3*f+1]==3*f+1)[0:c_f]]
        rd=[int(ind), int(ind)+1, int(ind)+2, int(ind)+3, int(ind)+4]
        index=torch.gather(frame_index[:,1,3*f+1],0,torch.Tensor(np.array(rd)).type(torch.long).to(device)).detach().cpu().numpy()
        #pdb.set_trace()
        c_f_let[cc].append(np.array([c_f]))
        #pdb.set_trace()
        corr_mat_let[cc].append(np.reshape(np.array([np.take(np.array(test_files),index)[0][2:7],np.take(np.array(test_files),index)[1][2:7],np.take(np.array(test_files),index)[2][2:7],np.take(np.array(test_files),index)[3][2:7],np.take(np.array(test_files),index)[4][2:7]]),(5,)))
        cc+=1
    #pdb.set_trace()
        # if j==10:
        #     pdb.set_trace()
#pdb.set_trace()
print("total Time taken: %.2fs" % (time.time() - start_time_1))
np.savetxt(Vis_dir+'corr_mat.txt', np.reshape(np.array(corr_mat_let),(-1,5)), delimiter=",", fmt="%s")
#pdb.set_trace()
np.savetxt(Vis_dir+'corr_mat_cf.txt', np.reshape(np.array(c_f_let),(-1,1)), delimiter=",", fmt="%s")
#np.savetxt(Vis_dir+'corr_mat.txt', frame_list.detach().cpu().numpy(), fmt='%i')
#np.savetxt(Vis_dir+'corr_mat_index.txt', frame_index.view(-1,2).detach().cpu().numpy(), fmt='%i')
#pdb.set_trace()