import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.backends import cudnn
from torch.utils.data import DataLoader

from torchvision import transforms

from core.resnet38_cls import Classifier

from tools.utils import *
from tools.txt_utils import *
from tools.dataset_utils import *
from tools.augment_utils import *
from tools.torch_utils import *
from tools.pickle_utils import *

###############################################################################
# Arguments
###############################################################################
parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--architecture', default='resnet38', type=str)

parser.add_argument('--la_crf', default=4, type=int)
parser.add_argument('--ha_crf', default=32, type=int)

parser.add_argument('--root_dir', default='F:/VOCtrainval_11-May-2012/', type=str)

parser.add_argument('--domain', default='val', type=str)
parser.add_argument('--model_name', default='VOC2012@arch=resnet38@pretrained=True@lr=0.1@wd=0.0005@bs=16@epoch=15', type=str)

args = parser.parse_args()
###############################################################################

if __name__ == '__main__':
    cam_dir = create_directory(f'./predictions/{args.model_name}/cam/')
    la_crf_dir = create_directory(f'./predictions/{args.model_name}/la_crf_{args.la_crf}/')
    ha_crf_dir = create_directory(f'./predictions/{args.model_name}/ha_crf_{args.ha_crf}/')

    class_names = ['background'] + read_txt('./data/class_names.txt')

    classes = len(class_names)
    class_dic = {class_name:class_index for class_index, class_name in enumerate(class_names)}

    dataset = VOC_Dataset_with_Mask(args.root_dir, f'./data/{args.domain}.txt', class_dic, None)
    
    cmap_dic, _ = get_color_map_dic('PASCAL_VOC')
    colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

    length = len(dataset)

    def get_gt(mask):
        h, w, c = mask.shape

        mask = mask.reshape((h * w, c))
        gt = np.zeros((h * w), dtype=np.int32)

        for class_index, color in enumerate(colors):
            color_mask = np.sum(np.abs(mask - color), axis=-1)
            gt[color_mask==0.] = class_index
            
            # color_mask = np.asarray(color_mask == 0., dtype=np.uint8) * 255
            # cv2.imshow(class_names[class_index], color_mask)

        # cv2.waitKey(0)
        return gt.reshape((h, w))

    def decode(data):
        h, w = data.shape
        image = np.zeros((h, w, 3), dtype = np.uint8)

        for y in range(h):
            for x in range(w):
                image[y, x, :] = colors[data[y, x]]

        return image

    def get_meanIU(pred, mask):
        inter = np.logical_and(pred, mask)
        union = np.logical_or(pred, mask)

        # cv2.imshow('inter', (inter*255).astype(np.uint8))
        # cv2.imshow('union', (union*255).astype(np.uint8))
        # cv2.waitKey(0)

        smooth = 1e-5
        miou = (np.sum(inter) + smooth) / (np.sum(union) + smooth)
        return miou

    mIoU_list = []
    mIoU_list_with_crf = []

    for i, (image_name, image, label, mask) in enumerate(dataset):
        sys.stdout.write('\r[{}/{}]'.format(i + 1, length))
        sys.stdout.flush()
        
        ori_image = np.asarray(image)
        h, w = ori_image.shape[:2]

        # 1. CAM
        cam_dict = load_pickle(cam_dir + image_name)

        cams = np.zeros((h, w, len(class_names)), dtype=np.float32)
        for class_index in range(len(class_names)):
            try:
                cams[..., class_index] = cam_dict[class_index] 
            except KeyError:
                pass

        # 2. CAM - low alpha
        la_cam_dict = load_pickle(la_crf_dir + image_name)

        la_cams = np.zeros((h, w, len(class_names)), dtype=np.float32)
        for class_index in range(len(class_names)):
            try:
                la_cams[..., class_index] = la_cam_dict[class_index]
            except KeyError:
                pass
        
        # 3. CAM - high alpha
        # ha_cam_dict = load_pickle(ha_crf_dir + image_name)

        # ha_cams = np.zeros((h, w, len(class_names)), dtype=np.float32)
        # for class_index in range(len(class_names)):
        #     try:
        #         ha_cams[..., class_index] = ha_cam_dict[class_index]
        #     except KeyError:
        #         pass

        bg = np.ones_like(cams[:, :, 0]) * 0.2
        bg = bg[..., np.newaxis]
        
        cams = np.argmax(np.concatenate([bg, cams], axis=-1), axis=-1)
        la_cams = np.argmax(la_cams, axis=-1)
        # ha_cams = np.argmax(ha_cams, axis=-1)
        
        gt = get_gt(mask)
        
        mIoU_list.append(get_meanIU(cams, gt))
        mIoU_list_with_crf.append(get_meanIU(la_cams, gt))

        # print(cams.shape)
        # print(la_cams.shape)
        # print(ha_cams.shape)

        # cv2.imshow('image', ori_image)
        # cv2.imshow('gt', decode(gt))
        # cv2.imshow('cams', decode(cams))
        # cv2.imshow('la_cams', decode(la_cams))
        # cv2.imshow('ha_cams', decode(ha_cams))
        # cv2.waitKey(0)
    
    print()

    print('# mIoU = {:.2f}%'.format(np.mean(mIoU_list) * 100))
    print('# mIoU with crf = {:.2f}%'.format(np.mean(mIoU_list_with_crf) * 100))

