import numpy as np
import cv2
import scipy

import skimage
from skimage import exposure
from skimage import color

from get_inst import *

import time

import matplotlib.pyplot as plt







def get_eval_inst(data_stuff, pcx, pcy, num_classes, pp, ii):

    imd, pxd, _, data_root, ID=get_inst(data_stuff, num_classes, pp, ii)

    my_im=(imd.copy()*255).astype(np.uint8);

    imd=cv2.resize(imd.astype(np.float32), (pcx, pcy))
    if np.amax(imd)>1:
        imd[::]=imd[::]/255.0;

    bx=np.expand_dims(np.expand_dims(imd, axis=0), axis=0)

    # RESIZE
    list_out=(bx, my_im, data_root, ID)

    return list_out





def return_overlay(imd, pxd): 

    imd_stack=np.stack((imd,imd,imd), axis=-1)
    #print(np.unique(pxd))

    colors=(
            (0,0,0),
            (255,0,0),
            (255,255,0),
            (0,255,0),
            (0,255,255),
            (0,0,255),
            (255,0,255),
            (153,50,204),
            (153,50,204),   
            (153,50,204),
            (153,50,204)
            )

    seg=np.zeros_like(imd_stack)

    for i in range(pxd.shape[0]):
        for j in range(pxd.shape[1]):
            seg[i,j,:]=colors[pxd[i,j]]
        
    ov=cv2.addWeighted(imd_stack, 0.7, seg, 0.3, 0)

    return ov













def connected_components(pxd_block, num_classes):

    num_comps=(1,1,1,2,1,1,1,1) # Number of connected components for each 
    pxd_cat=np.stack(pxd_block, axis=-1);
    #print(pxd_cat.shape)
    pxd_new=np.zeros_like(pxd_cat)
    
    for i in range(1, num_classes):
        pxd_temp=np.zeros_like(pxd_cat)
        idx=np.argwhere(pxd_cat==i)

        #print("idx shape")
        #print(idx.shape)
        
        pxd_temp[idx[:,0], idx[:,1], idx[:,2]]=1

        labeled, ncomponents = scipy.ndimage.measurements.label(pxd_temp)

        unique, counts = np.unique(labeled, return_counts=True) # Automatically returns sorted
        unique=unique[1:]; counts=counts[1:] # Get rid of first element --> background
        comps=dict(zip(unique, counts))
        #print("class " + str(i) + ": " + str(comps))
        ind = np.argpartition(counts, -num_comps[i-1])[-num_comps[i-1]:] # Returns indices to components with top volumes
        clusts=unique[ind] # labels of said components
        for jk in range(clusts.shape[0]):
            where_vec=np.asarray(np.where(np.asarray(labeled)==clusts[jk])) # Find pixels that belong to component of interest
            pxd_new[where_vec[0],where_vec[1],where_vec[2]]=i

    # Unpack back into list format
    pxd_new=np.split(pxd_new, pxd_new.shape[-1], -1)
    for i in range(len(pxd_new)):
        pxd_new[i]=np.squeeze(pxd_new[i])
    #print(pxd_new[0].shape)
    return pxd_new
















'''
def connected_components(x):

    labeled, ncomponents = scipy.ndimage.measurements.label(x)
    unique, counts = np.unique(labeled, return_counts=True) # Automatically returns sorted
    unique=unique[1:]; counts=counts[1:] # Get rid of first element --> background
    comps=dict(zip(unique,counts))

    return unique, labeled, comps
'''
'''
# Connected components
neighbor_structure=np.ones((3,3,3))
seg_new=np.zeros_like(hardmax)
# Apply SCAN clustering to each structure to get rid of false positive clusters
for mk in range(1,hardmax.shape[-1]): # EACH CLASS --> SKIP BACKGROUND

    current_im=hardmax[:,:,:,mk]
    labeled, ncomponents = scipy.ndimage.measurements.label(current_im)
    unique, counts = np.unique(labeled, return_counts=True) # Automatically returns sorted
    unique=unique[1:]; counts=counts[1:] # Get rid of first element --> background
    comps=dict(zip(unique,counts))
    ind = np.argpartition(counts, -custer_vec[mk])[-custer_vec[mk]:] # Returns indices to components with top volumes
    clusts=unique[ind] # labels of said components
    for jk in range(clusts.shape[0]):
        where_vec=np.asarray(np.where(np.asarray(labeled)==clusts[jk])) # Find pixels that belong to component of interest
        seg_new[where_vec[0],where_vec[1],where_vec[2],mk]=1

sums=np.sum(seg_new,axis=-1)
idx=np.asarray(np.where(sums==0)) # Indices to zeros
seg_new[idx[0],idx[1],idx[2],0]=1 # Classless voxels get assigned background class

def connected_components(x):

    labeled, ncomponents = scipy.ndimage.measurements.label(x)
    unique, counts = np.unique(labeled, return_counts=True) # Automatically returns sorted
    unique=unique[1:]; counts=counts[1:] # Get rid of first element --> background
    comps=dict(zip(unique,counts))

    return unique, labeled, comps
'''


















