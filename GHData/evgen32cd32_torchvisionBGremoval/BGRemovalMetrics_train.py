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
    return bstAnn;


class MyCocoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, root, annFile, transform):
        self.root = root;
        self.transform = transform;
        self.coco = COCO(annFile);
        self.ds = [];
        imgIds = self.coco.getImgIds();
        for i in imgIds:
            imgData = self.coco.loadImgs(i)[0];
            ann = bestAnn(self.coco,imgData);
            if ann is not None:
                if (ann['area']/(imgData['width']*imgData['height']) > 0.40):
                    self.ds.append((imgData,ann));
    
    def __getitem__(self, idx):
        imgData, ann = self.ds[idx];
        img = Image.open(self.root + imgData['file_name']);
        if self.transform is not None:
            img = self.transform(img);
        target = {};
        x,y,dx,dy = ann['bbox'];
        target["boxes"] = torch.as_tensor([x,y,x+dx,y+dy], dtype=torch.float32);
        target["labels"] = torch.as_tensor(1, dtype=torch.int64);
        target["masks"] = torch.as_tensor(self.coco.annToMask(ann), dtype=torch.uint8);
        target["image_id"] = torch.tensor([imgData['id']]);
        target["area"] = torch.as_tensor(ann['area'], dtype=torch.float32);
        target["iscrowd"] = torch.as_tensor(0, dtype=torch.int64);
        return img, [target];
    
    def __len__(self):
        return len(self.ds);

def main():
    
    path = './data/';
    ann_path = path + 'annotations_train_val2014/instances_train2014.json';
    img_path = path + 'train2014/';
    ann_path_val = path + 'annotations_train_val2014/instances_val2014.json';
    img_path_val = path + 'val2014/';
    save_path = './savestates/maskrcnn_resnet50_fpn_';
    metrics_train_path = './metrics/train_metrics_';
    
    testset = MyCocoDataset(root=img_path, annFile=ann_path, transform=transforms.ToTensor());
    
    print('Test dataset: ' + str(len(testset)));
    
    testloader = torch.utils.data.DataLoader(testset,shuffle=False);
    
    #dataiter = iter(testloader);
    #img, trg = dataiter.next();
    
    num_classes = 2;
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True);
    for param in model.parameters():
        param.requires_grad = False;
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features;
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes);
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels;
    hidden_layer = 256;
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes);
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu');
    model = model.to(device);
    
    
    num_epochs = 21;
    
    for i in range(11,num_epochs):
        now = datetime.now();
        current_time = now.strftime("%H:%M:%S");
        print(current_time +' epoch: ' + str(i));
        model.load_state_dict(torch.load(save_path + str(i)));
        model = model.eval();
        metrics = [];
        k = 0;
        pc = 0;
        with torch.no_grad():
            for images, targets in testloader:
                images = images.to(device);
                outputs = model(images);
                mask = (outputs[0]['masks'][outputs[0]['scores'].argmax()] >= 0.5);
                trgmask = targets[0]['masks'].bool().to(device);
                intersect = (mask*trgmask).sum().detach().cpu();
                union = (mask+trgmask).sum().detach().cpu();
                iou = intersect/union;
                img_id = targets[0]['image_id'][0][0];
                metrics.append([img_id.item(),iou.item()]);
                k += 1;
                if (np.floor(k*10/len(testloader)) > pc):
                    pc = np.floor(k*10/len(testloader));
                    now = datetime.now();
                    current_time = now.strftime("%H:%M:%S");
                    print(current_time + ' ' + str(int(pc)*10) + '%');
        df = pd.DataFrame(metrics,columns=['Image_Id','IoU']);
        df.to_csv(metrics_train_path + str(i) + '.csv',header=False, index=False);



if __name__ == "__main__":
    main();



