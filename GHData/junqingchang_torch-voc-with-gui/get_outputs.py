import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models

import numpy as np
from skimage import io
import os
import matplotlib.pyplot as plt
import json
import sys
from PIL import Image

VOC_DIRECTORY = 'train' # Directory ImageSet/Main
OUTPUT_DIRECTORY = 'goodfiles'
MODEL_DIRECTORY = 'goodfiles/bestresnetweights.pt'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class VOC2012VAL(Dataset):
    def __init__(self, voc_directory, difficult=False, transform=False):
        '''
        voc_directory: Directory of VOC, e.g. 'VOCdevkit/VOC2012'
        train_val: True for trainset, False for valset
        difficult: True if including difficult sets
        transform: Transformations
        '''
        self.train_val_directory = voc_directory + '/ImageSets/Main'
        self.image_directory = voc_directory + '/JPEGImages'
        self.difficult = difficult
        self.transform = transform
        files_in_directory = os.listdir(self.train_val_directory)
        assert 'val.txt' in files_in_directory
                    
        self.val_files = [x for x in files_in_directory if x.endswith('_val.txt') and not x.startswith('._')]
        self.val_files.sort()
        num_classes = len(self.val_files)
        self.val_onehot = {}
        self.idx_to_class = {}
        self.idx_to_file = {}
        ctr = 0
        vf = open(self.train_val_directory + '/val.txt', 'r')
        file_idx = 0
        for line in vf:
            self.val_onehot[line.strip()] = [0] * num_classes
            self.idx_to_file[file_idx] = line.strip()
            file_idx += 1
        for filename in self.val_files:
            self.idx_to_class[ctr] = filename[:-10]
            f = open(self.train_val_directory + '/' + filename, 'r')
            for line in f:
                fname, label = line.split()
                if label == '0' and self.difficult:
                    self.val_onehot[fname][ctr] = 1
                if label == '1':
                    self.val_onehot[fname][ctr] = 1
            ctr += 1
        removal = []
        for filename in self.val_onehot:
            if sum(self.val_onehot[filename]) == 0:
                removal.append(filename)
        for filename in removal:
            del(self.val_onehot[filename])
            
            
    def __len__(self):
        return len(self.val_onehot)
    
    def __getitem__(self, idx):
        filename = self.image_directory + '/' + self.idx_to_file[idx] + '.jpg'
        img = Image.open(filename)
        if self.transform:
            img = self.transform(img)
        target = torch.tensor(self.val_onehot[self.idx_to_file[idx]])
        return self.idx_to_file[idx], img, target.float()

def eval(model, device, test_loader):
    model.eval()
    test_loss = 0
    outputs = {}
    with torch.no_grad():
        for idx, (filename, data, target) in enumerate(test_loader):
            num_crop = data.shape[1]
            input_size = data.shape[0]
            data = data.view([-1,data.shape[-3],data.shape[-2],data.shape[-1]])
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.view(input_size, num_crop, -1)
            output = output.mean(1)
            output = torch.sigmoid(output)
            for i in range(len(filename)):
                output_target = []
                output_pred = []
                for j in range(len(target[i])):
                    output_target.append(target[i][j].tolist())
                    output_pred.append(output[i][j].tolist())
                outputs[filename[i]] = {'True': output_target, 'Pred': output_pred}
    return outputs



fivecrop_testtime = transforms.Compose([
        transforms.Resize(280),
        transforms.FiveCrop(224), 
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda norms: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(norm) for norm in norms]))
    ])

VOCVal = VOC2012VAL(VOC_DIRECTORY, transform = fivecrop_testtime)
val_loader = DataLoader(VOCVal, batch_size=4,
                        shuffle=False, num_workers=0)

resnet = models.resnet18(pretrained=False)
num_ftrs = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_ftrs, 20)
resnet.load_state_dict(torch.load(MODEL_DIRECTORY), strict=False)
resnet.to(device)

outputs = eval(resnet, device, val_loader)
with open('{}/evaloutputs.json'.format(OUTPUT_DIRECTORY), 'w') as fp:
    json.dump(outputs, fp)