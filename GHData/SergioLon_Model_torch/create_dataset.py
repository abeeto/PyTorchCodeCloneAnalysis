
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import pyvista as pv
import numpy as np
#import trimesh as tr
import networkx as nx
#import matplotlib.pyplot as plt
import torch
import copy
from torch_geometric.transforms import FaceToEdge,RandomRotate
from torch_geometric.data import Data,DataLoader,InMemoryDataset

def inputs_dataset(NUM_SAMPLES,file_path):

    #Dataset is a dict containing each input needed for the model
    
    #meshes is a dict containing all the meshes
    meshes={}
    faces={}
    d_list=[]
    #max_ver=0 #max number of points
    wss={}
    #max_wss=0
    dim=0 #the number of semples
    r_rot_x=RandomRotate(90,axis=0)
    r_rot_y=RandomRotate(90,axis=1)
    r_rot_z=RandomRotate(90,axis=2)
    for k in NUM_SAMPLES:
        #Miss the 5th and 14th mesh
        try:
            if k>10:
                mesh=pv.read(file_path+'/aorta_'+str(k)+'.vtp') #read the mesh
            else:
                mesh=pv.read(file_path+'/aorta_0'+str(k)+'.vtp')
            meshes[str(k)]=mesh#save the mesh in the dict
            faces[str(k)]=(mesh.faces.reshape((-1,4))[:, 1:4]+1).transpose()
            wss[str(k)]=mesh.point_arrays["WSS magnitude"]
            data=Data(x=torch.tensor(mesh.points,dtype=torch.float),
                      pos=torch.tensor(mesh.points,dtype=torch.float),
                  #edge_index=edge[j],
                      edge_attr=None,
                      y=None,
                      normal=None,
                      wss=torch.tensor(mesh.point_arrays["WSS magnitude"],dtype=torch.float),
                      face=torch.LongTensor(faces[str(k)])
                  )
            f2e=FaceToEdge(remove_faces=True)
            data=f2e(data)
        #print(data.edge_index)
            d_list.append(data)
            dim+=1
        except FileNotFoundError:
            continue
    d,slices=InMemoryDataset.collate(d_list)
    
    #n_d_list=[d_z,d_n_r]
    #d_new,slices_new=InMemoryDataset.collate(n_d_list)
    ##norm wss
    maxm = d.wss.max()
    minm = d.wss.min()
    wss_mean = ( maxm + minm ) / 2.
    wss_semidisp = ( maxm - minm ) / 2.
    
    ##
    #mean = np.mean(d_new.poss, axis=(0,1))
    maxm = d.pos.max(dim=-2).values
    minm = d.pos.min(dim=-2).values
    #print(minm)
    mean = ( maxm + minm ) / 2.
    for i in d_list:
        i.wss=(i.wss-wss_mean)/wss_semidisp
        i.pos = (i.pos - mean) / ( (maxm - minm) / 2 )
        i.x=i.pos
    #print(d_x.x)
    return DataLoader(d_list).dataset