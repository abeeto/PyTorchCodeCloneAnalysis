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


class MyCocoDataset(torch.utils.data.Dataset):
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
        target["masks"] = torch.as_tensor([self.coco.annToMask(ann)], dtype=torch.uint8);
        target['image_id'] =  torch.tensor([ann['image_id']]);
        x,y,dx,dy = ann['bbox'];
        target["boxes"] = torch.as_tensor([[x,y,x+dx,y+dy]], dtype=torch.float32);
        target['area'] = torch.as_tensor([ann['area']]);
        target['labels'] = torch.as_tensor([ann['category_id']]);
        target['iscrowd'] = torch.as_tensor([ann['iscrowd']]);
        return img, target;
    
    def __len__(self):
        return len(self.ds);


testset = MyCocoDataset(root=img_path_val, annFile=ann_path_val, transform=transforms.ToTensor());
testloader = torch.utils.data.DataLoader(testset,shuffle=False);
evaluate(model, testloader, device=device);


path = './data/';
ann_path = path + 'annotations_train_val2014/instances_train2014.json';
img_path = path + 'train2014/';
ann_path_val = path + 'annotations_train_val2014/instances_val2014.json';
img_path_val = path + 'val2014/';

trainset = MyCocoDataset(root=img_path, annFile=ann_path, transform=transforms.ToTensor());
testset = MyCocoDataset(root=img_path_val, annFile=ann_path_val, transform=transforms.ToTensor());
trainloader = torch.utils.data.DataLoader(trainset,shuffle=True);
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

params = [p for p in model.parameters() if p.requires_grad];
optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005);
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1);


train_one_epoch(model, optimizer, trainloader, device, i, print_freq=10);
evaluate(model, testloader, device=device);


#train
#train_one_epoch(model, optimizer, trainloader, device, i, print_freq=10);
#lr_scheduler.step();
#test
        



