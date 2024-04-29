# -*- coding: utf-8 -*-

"""
    @date: 2019.07.18
    @author: samuel ko
    @func: PRNet Training Part.
"""
import os
import cv2
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.optim
from model.resfcn256 import ResFCN256

from tools.WLP300dataset import PRNetDataset, ToTensor, ToNormalize
from tools.prnet_loss import WeightMaskLoss, INFO

from utils.utils import save_image, test_data_preprocess, make_all_grids, make_grid
from utils.losses import SSIM

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils, models
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

import argparse
import glob
import ast

import scipy.io as sio
from skimage import io
import skimage.transform

#画关键点
end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
def plot_kpt(image, kpt):
    ''' Draw 68 key points
    Args: 
        image: the input image (450,450,3)
        kpt: (68, 3).
    '''
    image = image.copy()
    kpt = np.round(kpt).astype(np.int32)
    for i in range(kpt.shape[0]):
        st = kpt[i, :2]
        image = cv2.circle(image,(st[0], st[1]), 1, (0,0,255), 2)  
        if i in end_list:
            continue
        ed = kpt[i + 1, :2]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)
    return image


FLAGS = {"start_epoch": 0,
         "target_epoch": 100,
         "device": "cuda",
         "mask_path": "./utils/uv_data/uv_weight_mask_gdh.png",
         "lr": 0.0001,
         "batch_size": 64, #16
         "save_interval": 1,
         "normalize_mean": [0.485, 0.456, 0.406],
         "normalize_std": [0.229, 0.224, 0.225],
         "images": "./results",
         "gauss_kernel": "original",
         "summary_path": "./prnet_runs",
         "summary_step": 0,
         "resume": True}

#函数的主要作用是：抠出人脸区域，并转换成(255,255,3)大小的图片
def process_image(image_path,image_h=255,image_w=255):

    image = cv2.imread(image_path)
    mat_path = image_path.replace('jpg', 'mat')
    info = sio.loadmat(mat_path)
    kpt = info['pt3d_68'] #(3,68)


    if image.shape[0] != 255 or image.shape[1] != 255:
        # 3. crop image with key points
        left = np.min(kpt[0, :])
        right = np.max(kpt[0,:])
        top = np.min(kpt[1,:])
        bottom = np.max(kpt[1, :])
        center = np.array([right - (right - left) / 2.0,
                            bottom - (bottom - top) / 2.0])
        old_size = (right - left + bottom - top) / 2
        size = int(old_size * 1.5)

        # random pertube. you can change the numbers
        marg = old_size * 0.1
        t_x = np.random.rand() * marg * 2 - marg
        t_y = np.random.rand() * marg * 2 - marg
        center[0] = center[0] + t_x
        center[1] = center[1] + t_y
        size = size * (np.random.rand() * 0.2 + 0.9)

        # crop and record the transform parameters
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, image_h - 1], [image_w - 1, 0]])
        tform = skimage.transform.estimate_transform('similarity', src_pts, DST_PTS)
        image = image / 255
        cropped_image = skimage.transform.warp(image, tform.inverse, output_shape=(image_h, image_w))

        '''
        image : 原始图片,通道格式是(b,g,r)
        cropped_image : 人脸区域图，图片大小是(255,255,3)
        tform : 原坐标系和目标坐标系(resize后的图片)之间的转换函数
        '''
        return image*255 , cropped_image *255 , kpt , tform
    
    else :
        print("error")
        return image , image , kpt , tform

def get_landmarks(pos,tform):
    '''
    pos : 模型预测的UV Map, (1,255,255,3)
    tform : 原图 和 resize图 坐标转换矩阵
    '''
    pos = pos.squeeze()
    pos_T = pos.transpose((1, 2, 0)) #(256,256,3)


    #根据tform参数，将关键点映射回原坐标系中(未resize)
    cropped_vertices = np.reshape(pos_T, [-1, 3]).T #(3,65536)
    z = cropped_vertices[2,:].copy()/tform.params[0,0]
    cropped_vertices[2,:] = 1
    vertices = np.dot(np.linalg.inv(tform.params), cropped_vertices) #(3,65536)
    vertices = np.vstack((vertices[:2,:], z))
    pos_T = np.reshape(vertices.T, [256, 256, 3])


    uv_kpt_ind = np.loadtxt('./uv_kpt_ind.txt').astype(np.int32)
    kpt = pos_T[uv_kpt_ind[1,:],uv_kpt_ind[0,:], :]
    return kpt

def cal_aflw2000_nme(model,image_folder,isShow=False):
    
    types = ('*.jpg', '*.png')
    image_path_list= []
    for files in types:
        image_path_list.extend(glob.glob(os.path.join(image_folder, files)))
    total_num = len(image_path_list)
    print(">>>一共有{}张测试图片".format(total_num))


    nme_list = []
    for i, image_path in tqdm(enumerate(image_path_list)):

        image_name = image_path.strip().split('/')[-1]

        #对输入的图片进行预处理
        src_image , crop_image , kpt , tform = process_image(image_path)
        image  = test_data_preprocess(crop_image)
        image_in = F.normalize(image, FLAGS["normalize_mean"], FLAGS["normalize_std"], False).unsqueeze_(0)

        #预测UV位置映射图
        pos = model(image_in).detach().cpu().numpy() *255

        
        if pos is None:
            print("------error------")
            continue

        #从UV位置映射图中抽取68个人脸关键点
        pred_kpt  = get_landmarks(pos,tform) #ndarray:(68,3),,得到的人脸关键点是基于未resize图的。
        gt_kpt = kpt.T #(68,3)

        #将关键点画出来并保存在图中
        if isShow:
            outputDir = 'Benchmark/AFLW2000_Result'
            plot_image = plot_kpt(src_image,pred_kpt) /255
            save_image_path = os.path.join(outputDir,image_name)
            io.imsave(save_image_path, np.squeeze(plot_image)[:,:,::-1])

        #计算 nme
        minx, maxx = np.min(gt_kpt[:, 0]), np.max(gt_kpt[:,0])
        miny, maxy = np.min(gt_kpt[:, 1]), np.max(gt_kpt[:,1])
        llength = np.sqrt((maxx - minx) * (maxy - miny))
        dis = pred_kpt[:,:2] - gt_kpt[:,:2]
        dis = np.sqrt(np.sum(np.power(dis,2),axis=1))
        dis = np.mean(dis)
        nme = dis / llength
        nme_list.append(nme)
    
    return np.mean(nme_list)

    

def main(args):

    # ---- init model
    model = ResFCN256()
    state = torch.load(args.model)
    model.load_state_dict(state['prnet'])
    model.to("cuda")

    # --- load data
    image_folder = args.inputDir
    save_folder = args.outputDir
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # --- cal nme
    mean_nme = cal_aflw2000_nme(model,image_folder,isShow=False)
    print("NME IS {}".format(mean_nme))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

    parser.add_argument('-i', '--inputDir', default='/home/beitadoge/Data/PRNet_PyTorch_Data/AFLW2000', type=str,
                        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--outputDir', default='Benchmark/AFLW2000_Result', type=str,
                        help='path to the output directory, where results(obj,txt files) will be stored.')
    parser.add_argument('-m', '--model', default='./results/latest.pth', type=str,
                        help='load model path')
    parser.add_argument('--isShow', default=False, type=ast.literal_eval,
                        help='whether to show the results with opencv(need opencv)')
    main(parser.parse_args())