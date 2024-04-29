import numpy as np
import os
import time
import matplotlib.pyplot as plt
import scipy
import cv2

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from net import *



    



class graph_class():

    def __init__(self, input_depth, num_classes, 
                        pcx, pcy, summ_step, 
                        save_step,  model_path,
                        ct=0, TB_ID=0, L_rate=0.0000001, 
                        cuda=0, load=False):

        # Lrate starts at 0.0000002
        self.input_depth=input_depth
        self.num_classes=num_classes
        self.pcx=pcx
        self.pcy=pcy
        self.L_rate=L_rate
        self.cuda=cuda
        
        self.display_step=summ_step
        self.save_step=save_step
        self.model_path=model_path
        self.ct=0
        self.TB_ID=TB_ID

        if self.cuda==1:
            self.device = torch.device("cuda:0")
        else:
            self.device=torch.device("cpu")
        
        if load==False:
            self.my_net=Net(self.input_depth, 
                            self.num_classes, 
                            self.device)
        else:
            self.load_model()
        
        self.my_loss=self.loss
        self.my_net.to(self.device)
        
        print('DEVICE INFO')
        print('CURRENT DEVICE: ' + str(self.device))
        print('CUDA AVAILABLE: ' + str(torch.cuda.is_available()))
        
        self.opt = torch.optim.Adam(self.my_net.parameters(), 
                                                lr=self.L_rate, 
                                                weight_decay=0.01)
        '''self.opt = torch.optim.RMSprop(self.my_net.parameters(), 
                                                lr=self.L_rate, 
                                                weight_decay=0.00001)'''
            
        if load==True:
            [];##self.load_opt()
        self.writer=SummaryWriter('runs/TB/' + str(TB_ID))
        return



    def update_pc(self, pcx, pcy):
        self.pcx=pcx;
        self.pcy=pcy;
        return






    def loss(self, seg, by, by_lm, by_mask, return_comps=False):
        
        values, indices = torch.max(by, 1)

        l_temp=F.cross_entropy(seg, indices, weight=torch.squeeze(by_lm))

        l=torch.mean(by_mask*l_temp)

        if return_comps==True:
            return l    
        else:
            return l



    def step_model(self, batch):

        t1=time.time()
        bx=torch.from_numpy(batch[0]).float().to(self.device)
        by=torch.from_numpy(batch[1]).float().to(self.device)
        by_lm=torch.from_numpy(batch[2]).float().to(self.device)
        by_mask=torch.from_numpy(batch[3]).long().to(self.device)

        # STEP
        self.opt.zero_grad()
        seg=self.my_net.forward(bx)
        #print(seg.size())
        loss_compute=self.my_loss(seg, by, by_lm, by_mask)
        loss_compute.backward()
        torch.nn.utils.clip_grad_norm_(self.my_net.parameters(), 5.0)
        self.opt.step()
        
        if self.ct%self.display_step==0 or self.ct<10:
            self.add_summary(bx, seg, by, by_lm, by_mask)
            print(str(time.time()-t1) + ' Seconds'); print('');
        if self.ct%self.save_step==0:
            self.save_model()
        if np.random.binomial(1, 0.02)==1:
            print(str(time.time()-t1) + ' Seconds'); print('');
        self.ct+=1
        return self.ct


    


    def save_model(self, path=None):
        if path==None:  
            path=self.model_path
        checkpoint = {'model': self.my_net,
                      'state_dict': self.my_net.state_dict(),
                      'optimizer' : self.opt.state_dict()}
                      #'scheduler' : self.sched.state_dict()}
        torch.save(checkpoint, self.model_path)         
        return
    
    

    def load_model(self, path=None):
        if path==None:  
            path=self.model_path

        print('Loading model ' + str(path)) 
        checkpoint = torch.load(path)
        self.my_net = checkpoint['model']
        self.my_net.load_state_dict(checkpoint['state_dict']) 

        self.my_net.eval()
        self.my_net.to(self.device)
        return  



    def load_opt(self, path=None):
        if path==None:  
            path=self.model_path

        print('Loading model ' + str(path)) 
        checkpoint = torch.load(path)

        self.opt.load_state_dict(checkpoint['optimizer'])  
        #self.sched.load_state_dict(checkpoint['scheduler'])

        return



    def add_summary(self, bx, seg, by, by_lm, by_mask):
        
        # Loss components
        l=self.loss(seg, by, by_lm, by_mask, return_comps=True) 
        self.writer.add_scalar('l', l, global_step=self.ct)
        
        print('Step ' + str(self.ct) + ' Loss: ' + str(l))

        # Original images
        grid1=torchvision.utils.make_grid(bx, nrow=2)
        self.writer.add_image('bx', grid1, global_step=self.ct)

        grid1=torchvision.utils.make_grid(by_mask, nrow=2)
        self.writer.add_image('loss_mask', grid1, global_step=self.ct)

        v,seg_max=torch.max(by, dim=1, keepdim=True)
        seg_max=seg_max.float()/self.num_classes;
        grid4=torchvision.utils.make_grid(seg_max, nrow=2)
        self.writer.add_image('seg_ground', grid4, global_step=self.ct)

        seg=F.softmax(seg, 1)
        v,seg_max=torch.max(seg, dim=1, keepdim=True)
        seg_max=seg_max.float()/self.num_classes;
        grid4=torchvision.utils.make_grid(seg_max, nrow=2)
        self.writer.add_image('seg_pred', grid4, global_step=self.ct)

        print('CURRENT LEARNING RATE: ' + str(self.opt.param_groups[0]['lr']))
        self.writer.add_scalar('learning_rate', self.opt.param_groups[0]['lr'], global_step=self.ct)

        return


    def get_prediction(self, b):
        
        t1=time.time()
        bx=torch.from_numpy(b[0]).float().to(self.device)
        my_im=b[1]
        h_o=my_im.shape[0]
        w_o=my_im.shape[1]
        
        seg=self.my_net.forward(bx);
        seg=seg.cpu().detach().numpy();
        seg=np.squeeze(seg); # Remove batch dim
        seg=np.transpose(seg, (1,2,0)) # Put channel dim at the end
        seg=cv2.resize(seg.astype(np.float32), (w_o, h_o)); # Resize to original image res

        # Argmax
        seg = np.squeeze(np.argmax(seg, axis=-1))

        return seg


    def connected_components(x):

        labeled, ncomponents = scipy.ndimage.measurements.label(x)
        unique, counts = np.unique(labeled, return_counts=True) 
        # Automatically returns sorted
        unique=unique[1:]; counts=counts[1:] # Get rid of first element --> background
        comps=dict(zip(unique,counts))

        return unique, labeled, comps

    def decode_all(imd_list, by, pcx,  pcy, num_classes):

        item_list=[];
        for i in range(by.shape[0]):
            items=decode_tensors(imd_list[i], by[i,:,:,:], pcx, pcy, num_classes)
            item_list.append(items)     
        return  item_list

    def decode_tensors(imd, seg, pcx, pcy, num_classes):

        h=imd.shape[0]   
        w=imd.shape[1]

        seg=np.transpose(seg, (1,2,0)) 
        seg=cv2.resize(seg.astype(np.float32), (w,h))

        # Segmentation maps are probability maps -- sigmoid 
        seg=np.round(seg)# argmax(seg, axis=-1)
        items_list=[];

        # Each class
        for i in range(num_classes): 
            
            temp=np.zeros((seg.shape[0], seg.shape[1]))
            temp[:,:]=seg[:,:,i]
            
            # Create connected components map  
            unique, labeled, comps=connected_components(temp)

            # Each predicted component for that class        
            for j in range(unique.shape[0]): 
                
                comp_map=np.zeros((seg.shape[0], seg.shape[1]))
                idx=np.argwhere(labeled==unique[j])
                
                comp_map[idx[:,0],idx[:,1]]=1
                clas=i

                # Box bounds            
                xmin=np.amin(idx[:,1])
                xmax=np.amax(idx[:,1])
                ymin=np.amin(idx[:,0])
                ymax=np.amax(idx[:,0])

                # Take mean confidence over segmentation map
                conf=np.mean(seg[idx[:,0],idx[:,1], i])

                # Size of component
                num_pix=comps[unique[j]]
               
                box=(clas, xmin, xmax, ymin, ymax, conf)
                item=(clas, comp_map, conf, num_pix, box, seg)
                items_list.append(item)

        return items_list

    def filter_items(item_list, num_classes):

        # Assume one object from each class
        # Therefore store most confident component from each class
        items_all=[];
        for i in range(len(item_list)): # Each image in batch
            items_temp=[]; # Keep track of retained items for that instance
            for j in range(num_classes): # Each class
                my_idx=None 

                for k in range(len(item_list[i])):
                    query_item=item_list[i][k]    
                    query_clas=query_item[0] 
     
                    if query_clas==j and my_idx==None:
                        my_idx=k   
                    # Keep largest component
                    elif query_clas==j and  query_item[2]>item_list[i][my_idx][2]:   
                        my_idx=k  
                            
                if my_idx is not None:
                    items_temp.append(item_list[i][my_idx])
            items_all.append(items_temp)
        return items_all

    def create_prediction_objects(im_list, by, filtered, pcx, pcy):

        pred_list=[]
        for i in range(len(im_list)):
            preds=all_predictions(im_list[i], by[i,:,:,:], filtered[i], pcx, pcy)
            pred_list.append(preds)
        return pred_list






