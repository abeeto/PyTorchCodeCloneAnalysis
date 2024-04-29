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
start_time = time.time()
img_number="/00138/"
data_main_path = './training_data/tiktok_data'
outpath = './correspondences/'
outpath_1 = './correspondences/corrs/'
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
    os.makedirs(outpath_1)

test_files = get_tiktok_data(data_main_path,num)
X_1 = np.zeros((1,IMAGE_HEIGHT,IMAGE_WIDTH,3,num),dtype='f')
for f in range(len(test_files)):
    data_name = test_files[f]
    DP = read_tiktok_data(data_main_path,data_name,IMAGE_HEIGHT,IMAGE_WIDTH)
    X_1[...,f]=DP

corr_mat_tik = np.genfromtxt(outpath +'corr_mat.txt',delimiter=",")
frame_num=np.shape(corr_mat_tik)[0]
pair_num=np.shape(corr_mat_tik)[1]-1
for i in range(frame_num):
    start_time_1 = time.time()
    for j in range(pair_num):
        start_time_2 = time.time()
        pass_point=0
        corres=[]
        corres_body=[]
        pdist = nn.PairwiseDistance(p=2)
        x_2=torch.Tensor(X_1[0,:,:,:,int(corr_mat_tik[i][0])-17633]).type(torch.float).to(device)
        x_2_1=torch.Tensor(X_1[0,:,:,:,int(corr_mat_tik[i][j+1])-17633]).type(torch.float).to(device)
        record_part_length=np.array(range(26))
        for k in range(24):
            body_part=0
            index=torch.where(x_2[...,2]==k+1)
            indicies=torch.stack([index[0],index[1]],dim=1)
            size=int(x_2[...,0][x_2[...,2]==k+1].size()[0])
            x_2_vu=torch.cat((x_2[...,0][x_2[...,2]==k+1].view(size,1),x_2[...,1][x_2[...,2]==k+1].view(size,1)),dim=1)
            record=torch.zeros_like(x_2_vu).type(torch.float).to(device)
            index_1=torch.where(x_2_1[...,2]==k+1)
            indicies_1=torch.stack([index_1[0],index_1[1]],dim=1)
            size_1=int(x_2_1[...,0][x_2_1[...,2]==k+1].size()[0])
            x_2_1_vu=torch.cat((x_2_1[...,0][x_2_1[...,2]==k+1].view(size_1,1),x_2_1[...,1][x_2_1[...,2]==k+1].view(size_1,1)),dim=1)
            for w in range(size):
                xx=torch.repeat_interleave(x_2_vu[w,:].view(1,2),size_1, dim=0)
                record[w,:]=x_2_vu[w,:]
                dupl=torch.sum(torch.unique(record, return_counts=True, dim=0)[1]>=2)==1
                #dupl=True
                if dupl==False:
                    record[w,:]=torch.tensor([0,0]).type(torch.float).to(device)
                else:
                    dist=pdist(xx,x_2_1_vu)
                    if np.shape(dist)[0]>=1:
                        shortest_dist=torch.topk(dist, 1, largest=False)
                        loc=indicies_1[shortest_dist[1][0],:].type(torch.float).to(device).view(1,2)
                        loc_1=indicies[w,:].type(torch.float).to(device).view(1,2)
                        #pdb.set_trace()
                        xy_dist=pdist(loc,loc_1)
                        #pdb.set_trace()
                    else:
                        xy_dist=1000
                        shortest_dist=[[1000]]
                    if xy_dist<7:
                        corres.append([])
                        res=np.concatenate((np.array([k+1]),np.array(indicies[w,:].detach().cpu().numpy())),axis=0)
                        res=np.concatenate((res,np.array(indicies_1[shortest_dist[1][0],:].detach().cpu().numpy())),axis=0)
                        corres[pass_point].append(res)
                        pass_point+=1
                        body_part+=1
            record_part_length[k+1]=pass_point
            if body_part<50:
                corres_body.append([])
                res=np.concatenate((np.array([k+1]),np.array([-1,-1])),axis=0)
                corres_body[k].append(res)
            else:
                corres_body.append([])
                res=np.concatenate((np.array([k+1]),np.array([record_part_length[k],record_part_length[k+1]-1])),axis=0)
                corres_body[k].append(res)
        np.savetxt(outpath_1+'00'+str(int(corr_mat_tik[i][0]))+'_'+'00'+str(int(corr_mat_tik[i][j+1]))+'_i_r1_c1_r2_c2.txt', np.reshape(np.array(corres),(-1,5)), delimiter=",", fmt="%s")
        np.savetxt(outpath_1+'00'+str(int(corr_mat_tik[i][0]))+'_'+'00'+str(int(corr_mat_tik[i][j+1]))+'_i_limit.txt', np.reshape(np.array(corres_body),(24,-1)), delimiter=",", fmt="%s")
        print('00'+str(int(corr_mat_tik[i][0]))+'_'+'00'+str(int(corr_mat_tik[i][j+1])))
        print("1 pair Time taken: %.2fs" % (time.time() - start_time_2))
    print("1 frame Time taken: %.2fs" % (time.time() - start_time_1))
print("total Time taken to get corrs: %.2fs" % (time.time() - start_time))
