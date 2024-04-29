import numpy as np
import cv2

import skimage
from skimage import exposure
from skimage import color

from get_inst import *

import time

import matplotlib.pyplot as plt









def get_batch(TOT_LIST):

    input_depth, batch_size, pc_x, pc_y, num_classes, data_stuff = TOT_LIST

    tt=time.time()
    bx=np.empty((batch_size, pc_y, pc_x, input_depth));
    by=np.empty((batch_size, pc_y, pc_x, num_classes));
    by_mask=np.empty((batch_size, pc_y, pc_x, 1))
    im_list=[]; pxd_list=[]; ID_list=[]; data_root_list=[];
    j=0;

    while j<batch_size:

        if True: #try:
            imd, pxd, lm, data_root, ID = get_inst(data_stuff, num_classes)
            
            imd, pxd, lm = augment(data_stuff, imd, pxd, lm)

            #print(np.unique(pxd))

            pxd=get_one_hot(pxd, num_classes)

            
            
            im_list.append((imd.copy()*255).astype(np.uint8));
            pxd_list.append(pxd) 
            ID_list.append(ID)
            data_root_list.append(ID)

            imd=cv2.resize(imd.astype(np.float32), (pc_x, pc_y))
            if np.amax(imd)>1:
                imd[::]=imd[::]/np.amax(imd);
            pxd=cv2.resize(pxd.astype(np.float32), (pc_x, pc_y))
            lm=cv2.resize(lm.astype(np.float32), (pc_x, pc_y))
            
            bx[j,:,:,0]=imd[:,:]
            by[j,:,:,:]=pxd[:,:,:] 
            by_mask[j,:,:,0]=lm[:,:]
            j+=1
                
        '''
        except Exception as e: 
            if np.random.binomial(1,0.05)==1:
                print(e); print('WARNING: BATCH ERROR');
            time.sleep(0.1)
        '''
                                  
    bx=np.transpose(bx, (0,3,1,2))    
    by=np.transpose(by, (0,3,1,2)) 
    by_mask=np.transpose(by_mask, (0,3,1,2)) 
    by_lm=get_loss_mask(by, num_classes)
    
    if np.isnan(bx).any():
        print("WARNING: NAN IN BX") 

    if np.isnan(by).any():
        print("WARNING: NAN IN BY") 

    if np.isinf(bx).any():
        print("WARNING: NAN IN BX") 

    if np.isinf(by).any():
        print("WARNING: NAN IN BY") 

    if np.random.binomial(1,0.05)==1:
        print("BATCH GEN TIME: " +str(time.time()-tt))
    
    list_out=(bx, by, by_lm, by_mask, im_list, pxd_list)
    return list_out





def get_loss_mask(by, num_classes):
    
    '''
    # pxd is in 1-hot
    lm=zeros(pxd.shape[0], pxd.shape[1])
    
    for i in range(pxd.shape[2]):
        div=np.sum(pxd[:,:,i]);
        temp=(1/div)*pxd[:,:,i]
        
        lm=lm+temp
    '''
    
    by_lm=np.zeros((1,by.shape[1],1,1))
    for i in range(num_classes):
        by_lm[0,i,0,0]=1#/(np.sum(by[:,i,:,:])**1.0+0.001)
    by_lm=by_lm*by.shape[2]*by.shape[3]*by.shape[0]
    #print(by_lm)
    return by_lm









def augment(data_stuff, imd, pxd, lm):
  
    # Gaussian noise
    if np.random.binomial(1,0.2)==1:
        sig=np.random.random()*0.1
        noise_mat=sig*np.random.randn(imd.shape[0], imd.shape[1])
        imd[:,:]+=noise_mat

    # Salt and pepper noise
    if np.random.binomial(1,0.2)==1:
        salt_param=np.random.random()*0.03
        pepp_param=np.random.random()*0.03
        salt_flag=np.random.binomial(1, salt_param, size=imd.shape)
        pepp_flag=np.random.binomial(1, pepp_param, size=imd.shape)
        imd[salt_flag==1]=1
        imd[pepp_flag==1]=0

    # Gaussian blur
    if np.random.binomial(1,0.2)==1:
        param=np.random.randint(low=1, high=4)*2+1
        imd=cv2.GaussianBlur(imd,(param,param),0)

    imd[imd<0]=0

    # Brightness
    if np.random.binomial(1, 0.2)==1:
        if np.random.binomial(1,0.5)==1:
            gam=np.random.random()*3+2
        else:
            gam=np.random.random()*0.3+0.15
        imd = exposure.adjust_gamma(imd, gamma=gam, gain=1)

    ##### Top
    if np.random.binomial(1, 0.1)==1:
        if np.random.binomial(1, 0.5)==1: # Pad
            pad_val=int(np.random.random()*0.2*imd.shape[0]);
            pad=np.mean(imd)*np.ones((pad_val, imd.shape[1]))
            pad2=np.zeros((pad_val, imd.shape[1]))
            imd=np.concatenate((pad, imd), axis=0)
            pxd=np.concatenate((pad2, pxd), axis=0)
            lm=np.concatenate((pad2, lm), axis=0)
        else: # Crop
            crop_val=int(np.random.random()*0.2*imd.shape[0]);
            imd=imd[crop_val:,:]
            pxd=pxd[crop_val:,:]
            lm=lm[crop_val:,:]

    ##### Bottom
    if np.random.binomial(1, 0.1)==1:
        if np.random.binomial(1, 0.5)==1: # Pad
            pad_val=int(np.random.random()*0.2*imd.shape[0]);
            pad=np.mean(imd)*np.ones((pad_val, imd.shape[1]))
            pad2=np.zeros((pad_val, imd.shape[1]))
            imd=np.concatenate((imd, pad), axis=0)
            pxd=np.concatenate((pxd, pad2), axis=0)
            lm=np.concatenate((lm, pad2), axis=0)
        else: # Crop
            crop_val=int(np.random.random()*0.2*imd.shape[0]);
            imd=imd[:(imd.shape[0]-crop_val),:]
            pxd=pxd[:(pxd.shape[0]-crop_val),:]
            lm=lm[:(lm.shape[0]-crop_val),:]

    ##### Left
    if np.random.binomial(1, 0.1)==1:
        if np.random.binomial(1, 0.5)==1: # Pad
            pad_val=int(np.random.random()*0.2*imd.shape[1]);
            pad=np.mean(imd)*np.ones((imd.shape[0], pad_val))
            pad2=np.zeros((imd.shape[0], pad_val))
            imd=np.concatenate((pad, imd), axis=1)
            pxd=np.concatenate((pad2, pxd), axis=1)
            lm=np.concatenate((pad2, lm), axis=1)
        else: # Crop
            crop_val=int(np.random.random()*0.2*imd.shape[0]);
            imd=imd[:,crop_val:]
            pxd=pxd[:,crop_val:]
            lm=lm[:,crop_val:]

    ##### Right
    if np.random.binomial(1, 0.1)==1:
        if np.random.binomial(1, 0.5)==1: # Pad
            pad_val=int(np.random.random()*0.2*imd.shape[1]);
            pad=np.mean(imd)*np.ones((imd.shape[0], pad_val))
            pad2=np.zeros((imd.shape[0], pad_val))
            imd=np.concatenate((imd, pad), axis=1)
            pxd=np.concatenate((pxd, pad2), axis=1)
            lm=np.concatenate((lm, pad2), axis=1)
        else: # Crop
            crop_val=int(np.random.random()*0.2*imd.shape[0]);
            imd=imd[:,:(imd.shape[1]-crop_val)]
            pxd=pxd[:,:(pxd.shape[1]-crop_val)] 
            lm=lm[:,:(lm.shape[1]-crop_val)] 

    if np.random.binomial(1, 0.2)==1:
        imd=np.amax(imd)-imd;
            
    return imd, pxd, lm



def get_one_hot(pxd, num_classes):

    pxd_one_hot=np.zeros((pxd.shape[0], pxd.shape[1], num_classes))
    for ii in range(pxd.shape[0]):
        for jj in range(pxd.shape[1]):
            pxd_one_hot[ii,jj, int(pxd[ii,jj])]=1

    return pxd_one_hot





