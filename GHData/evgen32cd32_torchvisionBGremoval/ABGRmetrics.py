import torch;
import torch.nn as nn;
import torch.nn.functional as F;
import torchvision;
import torchvision.transforms as transforms;
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor;
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor;

from engine import train_one_epoch, evaluate;
import utils;

from pycocotools.coco import COCO;

import numpy as np;
from PIL import Image;

import matplotlib.pyplot as plt;

import pandas as pd;
from datetime import datetime;

def barycenterWeight(an,w,h):
    box = an['bbox'];
    x = w/2 - (box[0] + box[2]/2);
    y = h/2 - (box[1] + box[3]/2);
    if (y == 0 and x == 0):
        return np.Inf;
    return an['area']/(x*x + y*y)**2;

def bestAnn(coco, imgData):
    annIds = coco.getAnnIds(imgIds=imgData['id'], catIds=[1]);
    anns = coco.loadAnns(annIds);
    bst = 0;
    bstAnn = None;
    for a in anns:
        if a['iscrowd'] == 1:
            continue;
        scr = barycenterWeight(a,imgData['width'],imgData['height']);
        if scr > bst:
            bst = scr;
            bstAnn = a;
    return (anns, bstAnn);


class MyCocoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, root, annFile, transform):
        self.root = root;
        self.transform = transform;
        self.coco = COCO(annFile);
        self.ds = [];
        imgIds = self.coco.getImgIds();
        for i in imgIds:
            imgData = self.coco.loadImgs(i)[0];
            anns, ann = bestAnn(self.coco,imgData);
            if ann is not None:
                if (ann['area']/(imgData['width']*imgData['height']) > 0.40):
                    self.ds.append((imgData,anns));
    
    def __getitem__(self, idx):
        imgData, anns = self.ds[idx];
        img = Image.open(self.root + imgData['file_name']);
        if self.transform is not None:
            img = self.transform(img);
        return img, anns;
    
    def __len__(self):
        return len(self.ds);


def bestOutput(output,h,w):
    bstMask = None;
    bst = 0;
    for i in range(len(output['boxes'])):
        if output['labels'][i] == 1:
            box = output['boxes'][i];
            x = w/2 - (box[0] + box[2])/2;
            y = h/2 - (box[1] + box[2])/2;
            mask = (output['masks'][i] > 0.5);
            if (y == 0 and x == 0):
                scr = np.Inf;
                return mask;
            else:
                scr = output['scores'][i] * mask.float().sum()/(x*x + y*y)**2
            if scr > bst:
                bst = scr;
                bstMask = mask;
    return bstMask;


def main():
	
    path = './data/';
    ann_path = path + 'annotations_train_val2014/instances_train2014.json';
    img_path = path + 'train2014/';
    ann_path_val = path + 'annotations_train_val2014/instances_val2014.json';
    img_path_val = path + 'val2014/';
    metrics_test_path = './metrics/ABGRtest_metrics';
    
    testset = MyCocoDataset(root=img_path_val, annFile=ann_path_val, transform=transforms.ToTensor());
    
    print('Test dataset: ' + str(len(testset)));
    
    testloader = torch.utils.data.DataLoader(testset,shuffle=False);
    
    num_classes = 2;
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True);
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu');
    model = model.to(device);
    model = model.eval();
    metrics = [];
    
    
    now = datetime.now();
    current_time = now.strftime("%H:%M:%S");
    print(current_time + ' start');
    k = 0;
    pc = 0;
    with torch.no_grad():
        for images, targets in testloader:
            images = images.to(device);
            #images = [images];
            outputs = model(images);
            img_id = targets[0]['image_id'].item();
            _, trgann = bestAnn(testset.coco,testset.coco.loadImgs(img_id)[0]);
            trgmask = torch.as_tensor(testset.coco.annToMask(trgann)).bool().to(device);
            mask = bestOutput(outputs[0],images[0][0].shape[0],images[0][0].shape[1]);
            #mask = (outputs[0]['masks'][outputs[0]['scores'].argmax()] >= 0.5);
            if mask is None:
                iou = 0;
            else:
                intersect = (mask*trgmask).sum().detach().cpu();
                union = (mask+trgmask).sum().detach().cpu();
                iou = (intersect/union).item();
            metrics.append([img_id,iou]);
            k += 1;
            if (np.floor(k*10/len(testloader)) > pc):
                pc = np.floor(k*10/len(testloader));
                now = datetime.now();
                current_time = now.strftime("%H:%M:%S");
                print(current_time + ' ' + str(int(pc)*10) + '%');
    df = pd.DataFrame(metrics,columns=['Image_Id','IoU']);
    df.to_csv(metrics_test_path + '.csv',header=False, index=False);


if __name__ == "__main__":
    main();



