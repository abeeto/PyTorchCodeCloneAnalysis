import argparse
import os
import numpy as np
import time
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset import *
from model import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--epoch', default=100, type = int)
parser.add_argument('--batch_size', default=32, type = int)
parser.add_argument('--aug', default=False, type = bool)
parser.add_argument('--lr', default=1e-4, type = float)
parser.add_argument('--txt', default=3, type = int)
args = parser.parse_args()
checkpoint_index = args.epoch
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test

class_num = 4 #cat dog person background

num_epochs = checkpoint_index
batch_size = args.batch_size


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True


if not args.test:
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320,augmentation=args.aug)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = optim.Adam(network.parameters(), lr = args.lr)
    # optimizer = optim.SGD(network.parameters(), lr = args.lr,momentum=0.9,weight_decay=0.0005)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        #TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_,img_name_,_,_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            loss_net.backward()
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1
        train_losses.append((avg_loss/avg_count).cpu().numpy())
        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        
        # if epoch % 10 == 9 or epoch % 10 == 4:
            #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        tmp = img_name_[0]
        file_name = "train_imgs/train_" + "%d"%(epoch+1)+'_'+tmp
        visualize_pred(file_name, pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        
        
        #VALIDATION
        network.eval()
        
        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate
        avg_loss_v = 0
        avg_count_v = 0
        
        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_,_,_,_ = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)
            loss_net_v = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            avg_loss_v += loss_net_v.data
            avg_count_v += 1
            
            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()
            
            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
        val_losses.append((avg_loss_v/avg_count_v).cpu().numpy())

        if epoch % 10 == 9 or epoch % 10 == 4:
            #visualize
            pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
            pred_box_ = pred_box[0].detach().cpu().numpy()
            file_name_val = "train_imgs/val_" + "%d"%(epoch+1)
            visualize_pred(file_name_val, pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        
        #optional: compute F1
        #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        #print(F1score)
        
        #save weights
        if epoch%10==9:
            #save last network
            print('saving net...')
            torch.save(network.state_dict(), 'checkpoints/network_%d.pth'%(epoch+1))
    plt.figure(1,figsize=(10,10))
    plt.xticks(np.arange(0,len(train_losses),50))
    plt.plot(train_losses,'b',label='Training data')
    plt.plot(val_losses,'r',label='Validation data')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('TrainError.png',dpi=400)
    plt.show()


else:
    #TEST
    # 3 means generate test .txt
    if args.txt == 3:
        dataset_test = COCO("data/test/images/", "", class_num, boxs_default, train = False, image_size=320)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    elif args.txt == 2: # 2 means val data generate .txt
        dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, image_size=320)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    elif args.txt == 1: # 1 means train data generate .txt
        dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, image_size=320)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    
    
    network.load_state_dict(torch.load('checkpoints/network_{}.pth'.format(str(checkpoint_index))))
    network.eval()
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, img_name_, height_, width_ = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        height_origin = height_[0].numpy()
        width_origin = width_[0].numpy()
        pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        write_txt(pred_box_, boxs_default, pred_confidence_, img_name_[0], height_origin, width_origin,args.txt)

        
        visualize_pred("test_imgs/"+'%d'%args.txt+'_'+img_name_[0], pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        cv2.waitKey(1000)



