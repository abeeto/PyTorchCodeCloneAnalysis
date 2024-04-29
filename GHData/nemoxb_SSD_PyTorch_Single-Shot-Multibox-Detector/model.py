import os
from pyexpat import model
import random
from turtle import forward
import numpy as np

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


def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    '''
    input:
    pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    
    output:
    loss            -- a single number for the value of the loss function

    explanation:
    For confidence (class labels), use cross entropy (F.cross_entropy)
    You can try F.binary_cross_entropy and see which loss is better
    For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)
    
    Note that you need to consider cells carrying objects and empty cells separately.
    I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    and reshape box to [batch_size*num_of_boxes, 4].
    Then you need to figure out how you can get the indices of all cells carrying objects,
    and use confidence[indices], box[indices] to select those cells.
    '''
    _,_,num_of_classes = pred_confidence.shape
    pred_confidence = pred_confidence.reshape((-1,num_of_classes))
    pred_box = pred_box.reshape((-1,4))
    ann_confidence = ann_confidence.reshape((-1,num_of_classes))
    ann_box = ann_box.reshape((-1,4))

    # indices_obj = (ann_confidence[:,-1]==0).nonzero().squeeze(1)
    # indices_noobj = (ann_confidence[:,-1]!=0).nonzero().squeeze(1)
    indices_obj = torch.where(ann_confidence[:,-1]==0)[0]
    indices_noobj = torch.where(ann_confidence[:,-1]!=0)[0]

    loss_cls_obj = F.cross_entropy(pred_confidence[indices_obj],ann_confidence[indices_obj])#,reduction='sum') #/ num_of_boxes
    loss_cls_noobj = F.cross_entropy(pred_confidence[indices_noobj],ann_confidence[indices_noobj])#,reduction='sum') #/ num_of_boxes
    loss_cls = loss_cls_obj + 3*loss_cls_noobj

    loss_box = F.smooth_l1_loss(pred_box[indices_obj],ann_box[indices_obj])# ,reduction='sum') #/ num_of_boxes

    loss = loss_cls + loss_box
    return loss

class ConvBatchReLU(nn.Module):
    '''
    purpose: 
    define a class for a recurrent computation consist of Convolution, BatchNorm and ReLU
    '''
    def __init__(self, cin, cout, k, s, pad=0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=cin,out_channels=cout,kernel_size=k,stride=s,padding=pad),
            nn.BatchNorm2d(cout),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class ConvReshape(nn.Module):
    '''
    purpose:
    define a class for last 4 divergent layers
    '''
    def __init__(self,ksize=3,pad=1):
        super().__init__()
        self.conv_box = nn.Conv2d(256, 16, ksize, 1, pad)
        self.conv_conf = nn.Conv2d(256, 16, ksize, 1, pad)
    
    def forward(self, x):
        out_b = self.conv_box(x)
        out_c = self.conv_conf(x)
        batch_size, channels, _, _ = out_b.shape
        out_b = out_b.reshape((batch_size, channels,-1))
        out_c = out_c.reshape((batch_size, channels,-1))
        return out_b, out_c

class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        # Conv blocks before divergence
        self.conv1 = ConvBatchReLU(3,64,3,2,1)
        self.conv2 = ConvBatchReLU(64,64,3,1,1)
        self.conv3 = ConvBatchReLU(64,64,3,1,1)
        self.conv4 = ConvBatchReLU(64,128,3,2,1)
        self.conv5 = ConvBatchReLU(128,128,3,1,1)
        self.conv6 = ConvBatchReLU(128,128,3,1,1)
        self.conv7 = ConvBatchReLU(128,256,3,2,1)
        self.conv8 = ConvBatchReLU(256,256,3,1,1)
        self.conv9 = ConvBatchReLU(256,256,3,1,1)
        self.conv10 = ConvBatchReLU(256,512,3,2,1)
        self.conv11 = ConvBatchReLU(512,512,3,1,1)
        self.conv12 = ConvBatchReLU(512,512,3,1,1)
        self.conv13 = ConvBatchReLU(512,256,3,2,1)

        # Divergence part
        self.diverge1 = ConvReshape()
        self.diverge2 = ConvReshape()
        self.diverge3 = ConvReshape()
        self.diverge4 = ConvReshape(ksize=1,pad=0)

        # Conv blocks between 1st and 2nd divergence part
        self.conv14 = ConvBatchReLU(256,256,1,1)
        self.conv15 = ConvBatchReLU(256,256,3,2,1)

        # Conv blocks between 2nd and 3rd divergence part
        self.conv16 = ConvBatchReLU(256,256,1,1)
        self.conv17 = ConvBatchReLU(256,256,3,1)

        # Conv blocks between 3rd divergence part and main output part
        self.conv18 = ConvBatchReLU(256,256,1,1)
        self.conv19 = ConvBatchReLU(256,256,3,1)

        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        # x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        # 10*10 boxes output
        out_b10, out_c10 = self.diverge1(x)

        x = self.conv14(x)
        x = self.conv15(x)
        # 5*5 boxes output
        out_b5, out_c5 = self.diverge2(x)

        x = self.conv16(x)
        x = self.conv17(x)
        # 3*3 boxes output
        out_b3, out_c3 = self.diverge3(x)

        x = self.conv18(x)
        x = self.conv19(x)
        # 1*1 boxes output
        out_b1, out_c1 = self.diverge4(x)

        bboxes = torch.cat([out_b10,out_b5,out_b3,out_b1],dim=2)
        confidence = torch.cat([out_c10,out_c5,out_c3,out_c1],dim=2)

        bboxes = torch.permute(bboxes,[0,2,1])  
        confidence = torch.permute(confidence,[0,2,1])    
        
        batch_size,_,_ = bboxes.shape
        bboxes = bboxes.reshape([batch_size,-1,4])  
        confidence = confidence.reshape([batch_size,-1,self.class_num])
        confidence = F.softmax(confidence,dim=2)
        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        
        return confidence, bboxes

if __name__ == '__main__':
    dummy = torch.randn(4, 3,320,320, requires_grad=True)
    model_test = SSD(4)
    confidence, bboxes= model_test(dummy)
    print(bboxes.shape)
    print(confidence.shape)