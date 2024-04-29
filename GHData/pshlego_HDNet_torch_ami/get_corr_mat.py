from os import path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pdb
import torch
import torch.nn as nn
import time
from utils import (get_origin_scaling,get_concat_h, depth2mesh, read_test_data, nmap_normalization, get_test_data,get_tiktok_data,read_tiktok_data) 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())
start_time_1 = time.time()
img_number="/00138/"
data_main_path = './training_data/tiktok_data'
outpath = './correspondences/'
visualization = True
num=306
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
def write_matrix_txt(a,filename):
    mat = np.matrix(a)
    with open(filename,'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.1f')

if not path.exists(outpath):
    print("outpath created!")
    os.makedirs(outpath)

test_files = get_tiktok_data(data_main_path,num)
X_1 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3,num),dtype='f')
for f in range(len(test_files)):
    data_name = test_files[f]
    DP = read_tiktok_data(data_main_path,data_name,IMAGE_HEIGHT,IMAGE_WIDTH)
    X_1[...,f]=DP
cc=0
frame_list=torch.zeros((num,num)).type(torch.int).to(device)
frame_index=torch.zeros((num,2,num)).type(torch.int).to(device)
x_2=torch.Tensor(X_1[0,:,:,:,1]).type(torch.int).to(device)
corr_mat_let=[]
for f in range(int(num/3)):
    start_time = time.time()
    c_f=0# correspondence가 일정 개수 이상 overapping된 body part가 5개 이상인 framework
    x_2=torch.Tensor(X_1[0,:,:,:,3*f+1]).type(torch.int).to(device)
    for j in range(int(num/3)):
        x_2_1=torch.Tensor(X_1[0,:,:,:,3*j+1]).type(torch.int).to(device)
        X_wo_z_1=x_2.view(256*256,3)
        X_wo_z_2=x_2_1.view(256*256,3)
        X_count_1=torch.unique(X_wo_z_1, return_counts=True, dim=0)#각 파트에서도 (V,U,I)쌍이 중복되는 것을 제외함
        X_count_2=torch.unique(X_wo_z_2, return_counts=True, dim=0)#각 파트에서도 (V,U,I)쌍이 중복되는 것을 제외함
        X_wo_z=torch.cat((X_count_1[0],X_count_2[0]),dim=0)
        X_count=torch.unique(X_wo_z, return_counts=True, dim=0)
        X_final=torch.where(X_count[1]==2, X_count[0][:,2], torch.zeros_like(X_count[0][:,2]).type(torch.int).to(device)).type(torch.int).to(device)
        count=torch.sum(torch.unique(torch.sort(X_final)[0],return_counts=True)[1]>=30)
        if count>=5:
            frame_list[3*f+1,3*j+1]=1
            frame_index[c_f,:,3*f+1]=torch.Tensor([3*f+1,3*j+1]).type(torch.int).to(device)
            c_f+=1
    print("1 frame Time taken: %.2fs" % (time.time() - start_time))
    frame_index[c_f:c_f*2-1,:,3*f+1]=torch.from_numpy(np.flip(frame_index[0:c_f-1,:,3*f+1].detach().cpu().numpy(),axis=0).copy()).to(device)
    if c_f>=6:
        corr_mat_let.append([])
        lf=torch.arange(c_f)
        ind=lf[(frame_index[:,1,3*f+1]==3*f+1)[0:c_f]]
        rd=[int(ind), int(ind)+1, int(ind)+2, int(ind)+3, int(ind)+4]
        index=torch.gather(frame_index[:,1,3*f+1],0,torch.Tensor(np.array(rd)).type(torch.long).to(device)).detach().cpu().numpy()
        corr_mat_let[cc].append(np.reshape(np.array([np.take(np.array(test_files),index)[0][2:7],np.take(np.array(test_files),index)[1][2:7],np.take(np.array(test_files),index)[2][2:7],np.take(np.array(test_files),index)[3][2:7],np.take(np.array(test_files),index)[4][2:7]]),(5,)))
        cc+=1
print("create corr_mat Time taken: %.2fs" % (time.time() - start_time_1))
np.savetxt(outpath+'corr_mat.txt', np.reshape(np.array(corr_mat_let),(-1,5)), delimiter=",", fmt="%s")