import os
import re
import sys
import cv2
import math
import time
import scipy
import argparse
import matplotlib
#from torch import np
import numpy as np
import pylab as plt
from joblib import Parallel, delayed
import util
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
import skimage.morphology
from skimage.morphology import square, dilation, erosion

#parser = argparse.ArgumentParser()
#parser.add_argument('--t7_file', required=True)
#parser.add_argument('--pth_file', required=True)
#args = parser.parse_args()


#device = torch.device("cuda:1")
torch.set_num_threads(torch.get_num_threads())
weight_name = '/home/yiming/code/data/model/pose_model.pth'

blocks = {}


limb_seq = [[1,2], [2,3], [3,4], [4,5], [2,6], [6,7], [7,8], [2,9], [9,10], [10,11], [9,3], [2,12], [12,13], [13,14],
           [9,12], [6,12], [2,17], [2,18], [1,17], [1,18], [1,15], [1,16], [15,17], [16,18]]

             
block0  = [{'conv1_1':[3,64,3,1,1]},{'conv1_2':[64,64,3,1,1]},{'pool1_stage1':[2,2,0]},{'conv2_1':[64,128,3,1,1]},{'conv2_2':[128,128,3,1,1]},{'pool2_stage1':[2,2,0]},{'conv3_1':[128,256,3,1,1]},{'conv3_2':[256,256,3,1,1]},{'conv3_3':[256,256,3,1,1]},{'conv3_4':[256,256,3,1,1]},{'pool3_stage1':[2,2,0]},{'conv4_1':[256,512,3,1,1]},{'conv4_2':[512,512,3,1,1]},{'conv4_3_CPM':[512,256,3,1,1]},{'conv4_4_CPM':[256,128,3,1,1]}]

blocks['block1_1']  = [{'conv5_1_CPM_L1':[128,128,3,1,1]},{'conv5_2_CPM_L1':[128,128,3,1,1]},{'conv5_3_CPM_L1':[128,128,3,1,1]},{'conv5_4_CPM_L1':[128,512,1,1,0]},{'conv5_5_CPM_L1':[512,38,1,1,0]}]

blocks['block1_2']  = [{'conv5_1_CPM_L2':[128,128,3,1,1]},{'conv5_2_CPM_L2':[128,128,3,1,1]},{'conv5_3_CPM_L2':[128,128,3,1,1]},{'conv5_4_CPM_L2':[128,512,1,1,0]},{'conv5_5_CPM_L2':[512,19,1,1,0]}]

for i in range(2,7):
    blocks['block%d_1'%i]  = [{'Mconv1_stage%d_L1'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L1'%i:[128,128,7,1,3]},
{'Mconv5_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L1'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L1'%i:[128,38,1,1,0]}]
    blocks['block%d_2'%i]  = [{'Mconv1_stage%d_L2'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L2'%i:[128,128,7,1,3]},
{'Mconv5_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L2'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L2'%i:[128,19,1,1,0]}]

#get pose encoded with heatmap from coordinates of the joints
def reverse_xy(raw_key_points):
    new_key_points = []
    #reverse x and y coordinate
    for item in raw_key_points:
        if item != []:
            new_point = (item[1], item[0])
        else:
            new_point = []        
        new_key_points.append(new_point)
    return new_key_points

#fill in the radius of 4
def fill_in(pose_img, point, radius):
    
    #pose_img = np.zeros([256, 256])
    point_x, point_y = point
    for x in range(point_x - radius, point_x + radius + 1):
        for y in range(point_y - radius, point_y + radius + 1):
            if ((x-point_x)*(x-point_x)+(y-point_y)*(y-point_y)) <= 16:
                if (x >= 0) and (x <= 255) and (y >= 0) and (y <= 255): 
                    pose_img[x][y] = 255
    return pose_img
    

def fill_in2(pose_img, point, radius):
    
    #pose_fill_img = np.zeros([256, 256])
    pose_fill_img = pose_img
    point_x, point_y = point
    for x in range(point_x - radius, point_x + radius + 1):
        for y in range(point_y - radius, point_y + radius + 1):
            if ((x-point_x)*(x-point_x)+(y-point_y)*(y-point_y)) <= 16:
                if (x >= 0) and (x <= 255) and (y >= 0) and (y <= 255): 
                    pose_fill_img[x][y] = 255
    return pose_fill_img

def get_heatmap_pose(new_key_points):
    #print ('Get Heatmap...')
    heatmap_pose = np.zeros([256, 256])
    for item in new_key_points:
        if item != []:
            heatmap_pose = fill_in2(heatmap_pose, item, 4)
    return heatmap_pose

def get_18_heatmaps(new_key_points):
    heatmaps_18 = []
    for item in new_key_points:
        heatmap = np.zeros([256, 256])
        if (item != []):
            heatmap = fill_in(heatmap, item, 4)
        heatmaps_18.append(heatmap)
    return heatmaps_18

#connect corresponding joints together
def connect_keypoints(pose_img, new_key_points, limb_seq):
    #print ('Get Skeleton...') 
    fill_list = []
    for i in range (len(limb_seq)):
        #fill_list = []
        #get the equation of the line
        point1 = new_key_points[limb_seq[i][0] - 1]
        #print(point1)
        point2 = new_key_points[limb_seq[i][1] - 1]
        #print(point2)
        if (point1 != []) and (point2 != []):
            x1, y1 = point1
            x2, y2 = point2
            if x2 - x1 == 0:
                if (y1 < y2):
                    for y in range(y1, y2):
                        fill_list.append([x1, y])
                else:
                    for y in range(y2, y1):
                        fill_list.append([x1, y])
            else:
                k = (y2 - y1)/(x2 - x1)
                b = y1 - k*x1
                if (x1 < x2):
                    #for x in range(x1, x2):
                    xs = np.linspace(x1, x2, 50)    
                    for x in xs: 
                        fill_list.append([int(x), int(k*x + b)])
                else:
                    #for x in range(x2, x1):
                    xs = np.linspace(x2, x1, 50)
                    for x in xs:
                        fill_list.append([int(x), int(k*x + b)])

    for item in fill_list:
        skeleton = fill_in(pose_img, item, 4)
    
    return skeleton

def get_connect_list(pose_img, new_key_points, limb_seq):
    #check the sequence of connection
    connect_list = []
    for i in range (len(limb_seq)):
        fill_list = []
        #get the equation of the line
        point1 = new_key_points[limb_seq[i][0] - 1]
        print(point1)
        point2 = new_key_points[limb_seq[i][1] - 1]
        print(point2)
        if (point1 != []) and (point2 != []):
            x1, y1 = point1
            x2, y2 = point2
            if x2 - x1 == 0:
                if (y1 < y2):
                    for y in range(y1, y2):
                        fill_list.append([x1, y])
                else:
                    for y in range(y2, y1):
                        fill_list.append([x1, y])
            else:
                k = (y2 - y1)/(x2 - x1)
                b = y1 - k*x1
                if (x1 < x2):
                    #for x in range(x1, x2):
                    xs = np.linspace(x1, x2, 50)    
                    for x in xs: 
                        fill_list.append([int(x), int(k*x + b)])
                else:
                    #for x in range(x2, x1):
                    xs = np.linspace(x2, x1, 50)
                    for x in xs:
                        fill_list.append([int(x), int(k*x + b)])
	    
	    print (fill_list)

            for item in fill_list:
                connect_item = fill_in2(pose_img, item, 4)
            connect_list.append(connect_item)

    return connect_list


def get_mask(skeleton_pose):
    #print('Get the mask...')
    return erosion(dilation(skeleton_pose, square(25)), square(10))

def make_layers(cfg_dict):
    layers = []
    for i in range(len(cfg_dict)-1):
        one_ = cfg_dict[i]
        for k,v in one_.iteritems():      
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    one_ = cfg_dict[-1].keys()
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)
    
layers = []
for i in range(len(block0)):
    one_ = block0[i]
    for k,v in one_.iteritems():      
        if 'pool' in k:
            layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
            layers += [conv2d, nn.ReLU(inplace=True)]  
       
models = {}           
models['block0']=nn.Sequential(*layers)        

for k,v in blocks.iteritems():
    models[k] = make_layers(v)
                
class pose_model(nn.Module):
    def __init__(self,model_dict,transform_input=False):
        super(pose_model, self).__init__()
        self.model0   = model_dict['block0']
        self.model1_1 = model_dict['block1_1']        
        self.model2_1 = model_dict['block2_1']  
        self.model3_1 = model_dict['block3_1']  
        self.model4_1 = model_dict['block4_1']  
        self.model5_1 = model_dict['block5_1']  
        self.model6_1 = model_dict['block6_1']  
        
        self.model1_2 = model_dict['block1_2']        
        self.model2_2 = model_dict['block2_2']  
        self.model3_2 = model_dict['block3_2']  
        self.model4_2 = model_dict['block4_2']  
        self.model5_2 = model_dict['block5_2']  
        self.model6_2 = model_dict['block6_2']
        
    def forward(self, x):    
        out1 = self.model0(x)
        
        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2  = torch.cat([out1_1,out1_2,out1],1)
        
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3   = torch.cat([out2_1,out2_2,out1],1)
        
        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4   = torch.cat([out3_1,out3_2,out1],1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5   = torch.cat([out4_1,out4_2,out1],1)  
        
        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6   = torch.cat([out5_1,out5_2,out1],1)         
              
        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)
        
        return out6_1,out6_2        


model = pose_model(models)     
model.load_state_dict(torch.load(weight_name))
model.cuda()
model.float()
model.eval()

param_, model_ = config_reader()

#torch.nn.functional.pad(img pad, mode='constant', value=model_['padValue'])
#for imgs in os.path()

img = '/home/yiming/code/data/DeepFashion/DF_img_pose/test_samples_img/00038_7.jpg'
oriImg = cv2.imread(img) # B,G,R order

multiplier = [x * model_['boxsize'] / oriImg.shape[0] for x in param_['scale_search']]
heatmap_avg = torch.zeros((len(multiplier),19,oriImg.shape[0], oriImg.shape[1])).cuda()
paf_avg = torch.zeros((len(multiplier),38,oriImg.shape[0], oriImg.shape[1])).cuda()

for m in range(len(multiplier)):
    scale = multiplier[m]
    h = int(oriImg.shape[0]*scale)
    w = int(oriImg.shape[1]*scale)
    pad_h = 0 if (h%model_['stride']==0) else model_['stride'] - (h % model_['stride']) 
    pad_w = 0 if (w%model_['stride']==0) else model_['stride'] - (w % model_['stride'])
    new_h = h+pad_h
    new_w = w+pad_w

    imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_['stride'], model_['padValue'])
    imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5

    feed = Variable(T.from_numpy(imageToTest_padded)).cuda()      
    output1,output2 = model(feed)

    heatmap = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])).cuda()(output2)

    paf = nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])).cuda()(output1)       

    heatmap_avg[m] = heatmap[0].data
    paf_avg[m] = paf[0].data  



heatmap_avg = T.transpose(T.transpose(T.squeeze(T.mean(heatmap_avg, 0)),0,1),1,2).cuda() 
heatmap_avg=heatmap_avg.cpu().numpy()
paf_avg    = paf_avg.cpu().numpy()


all_peaks = []
peak_counter = 0

#maps = 
for part in range(18):
    map_ori = heatmap_avg[:,:,part]
    map = gaussian_filter(map_ori, sigma=3)

    map_left = np.zeros(map.shape)
    map_left[1:,:] = map[:-1,:]
    map_right = np.zeros(map.shape)
    map_right[:-1,:] = map[1:,:]
    map_up = np.zeros(map.shape)
    map_up[:,1:] = map[:,:-1]
    map_down = np.zeros(map.shape)
    map_down[:,:-1] = map[:,1:]

    peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param_['thre1']))
#    peaks_binary = T.eq(
#    peaks = zip(T.nonzero(peaks_binary)[0],T.nonzero(peaks_binary)[0])

    peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse

    peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
    id = range(peak_counter, peak_counter + len(peaks))
    peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

    all_peaks.append(peaks_with_score_and_id)
    peak_counter += len(peaks)

raw_key_points = []
for item in all_peaks:
    if item != []:
        raw_key_points.append(item[0][0:2])
    else:
        raw_key_points.append([])

new_key_points = reverse_xy(raw_key_points)

'''
heatmap_pose_path = os.path.join(root_path, 'test_samples_img_heatmap/')
if(os.path.exists(heatmap_pose_path) is False):
    os.mkdir(heatmap_pose_path)

skeleton_pose_path = os.path.join(root_path, 'test_samples_img_skeleton/')
if (os.path.exists(skeleton_pose_path) is False):
    os.mkdir(skeleton_pose_path)

pose_mask_path = os.path.join(root_path, 'test_samples_img_mask/')
if (os.path.exists(pose_mask/_path) is False):
    os.mkdir(pose_mask_path)
'''

heatmap_pose = get_heatmap_pose(new_key_points)
heatmaps_18 = get_18_heatmaps(new_key_points)
connect_list = get_connect_list(heatmap_pose, new_key_points, limb_seq)

for i in range(len(heatmaps_18)):
    cv2.imwrite(os.path.join('/home/yiming/code/data/DeepFashion/DF_img_pose/debug/', ('heatmap' + '_' + str(i+1) + '.jpg')), heatmaps_18[i])

cv2.imwrite(os.path.join('/home/yiming/code/data/DeepFashion/DF_img_pose/debug/', 'heatmap.jpg'), heatmap_pose)



skeleton_pose = connect_keypoints(heatmap_pose, new_key_points, limb_seq)
cv2.imwrite(os.path.join('/home/yiming/code/data/DeepFashion/DF_img_pose/debug/', ('skeleton' + '.jpg')), skeleton_pose)

pose_mask = get_mask(skeleton_pose)
cv2.imwrite(os.path.join('/home/yiming/code/data/DeepFashion/DF_img_pose/debug/', ('mask' + '.jpg')), pose_mask)


print (len(connect_list))
for i in range(len(connect_list)):
    cv2.imwrite(os.path.join('/home/yiming/code/data/DeepFashion/DF_img_pose/debug/', ('connect' + '_' + str(i+1) + '.jpg')), connect_list[i])




