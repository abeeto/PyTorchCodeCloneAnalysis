# William Burton
import numpy as np
import imageio
import os
import cv2
import time
import matplotlib.pyplot as plt
import random
import math








def get_inst(data_stuff, 
                num_classes, 
                pool_num=None, 
                ct=None):
                
    imd, pxd, lm, data_root, ID = import_comps(data_stuff, num_classes, pool_num=pool_num, ct=ct)
    return imd, pxd, lm, data_root, ID
    






def import_comps(data_stuff, num_classes, pool_num=None, ct=None):

    lm=[];

    # Choose instance   
    data_root, ID, lm_coeff=data_stuff.sample_sup_ID(pool_num=pool_num, ct=ct)
    image_ext='.png'

    #if data_stuff.train_mode==False:
    #    image_ext='.jpg'

    imd_path=data_root + '/Imd/' + ID + image_ext
    imd=imageio.imread(imd_path); 

    if data_stuff.train_mode==True:
        pxd_path=data_root + '/Pxd/' + ID + image_ext
        pxd=imageio.imread(pxd_path);
    else:
        #imd=imd[:,:,0]
        pxd=[];
    
    # Correct class numbers
    if data_stuff.train_mode==True:
        if data_root=='/home/will/Desktop/OLD/Multiclass/Train/Sagittal/':
            # Patella is background
            idx=np.argwhere(pxd==3);
            if idx.shape[0]>0:
                pxd[idx[:,0], idx[:,1]]=0;

            # Patcart is background
            idx=np.argwhere(pxd==4);
            if idx.shape[0]>0:
                pxd[idx[:,0], idx[:,1]]=0;

            # Tibia is 3
            idx=np.argwhere(pxd==5);
            if idx.shape[0]>0:
                pxd[idx[:,0], idx[:,1]]=3;

            # Tibia is 4
            idx=np.argwhere(pxd==6);
            if idx.shape[0]>0:
                pxd[idx[:,0], idx[:,1]]=4;

        # Loss coeffs
        if lm_coeff==1:
            lm=np.ones_like(pxd)
        else: 
            lm=np.ones_like(pxd)
            idx=np.argwhere(pxd==0)
            lm[idx[:,0], idx[:,1]]=0
    
    return imd/255, pxd, lm, data_root, ID
    
 





