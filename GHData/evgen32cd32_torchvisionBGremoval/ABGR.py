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

def main():
	
	path = './data/';
	ann_path = path + 'annotations_train_val2014/instances_train2014.json';
	img_path = path + 'train2014/';
	ann_path_val = path + 'annotations_train_val2014/instances_val2014.json';
	img_path_val = path + 'val2014/';
	
	testset = MyCocoDataset(root=img_path_val, annFile=ann_path_val, transform=transforms.ToTensor());
	
	print('Test dataset: ' + str(len(testset)));
	
	testloader = torch.utils.data.DataLoader(testset,shuffle=False);
	
	#dataiter = iter(testloader);
	#img, trg = dataiter.next();
	
	num_classes = 2;
	model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True);
	
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu');
	model = model.to(device);
	
	num_epochs = 1;
	
	evaluate(model, testloader, device=device);


if __name__ == "__main__":
    main();



