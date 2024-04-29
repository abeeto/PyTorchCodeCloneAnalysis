import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
from skimage import io
import os
import matplotlib.pyplot as plt
import json
import sys
from PIL import Image



VOC_DIRECTORY = 'VOCdevkit/VOC2012' # Directory ImageSet/Main where VOC_DIRECTORY/ImageSet/Main contains train.txt
OUTPUT_DIRECTORY = 'coloradjust' # Directory where per epoch data/model is stored
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
start_epoch = 1
epochs = 50
learning_rate = 0.01
momentum = 0.1
log_freq = 20
threshold = [0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55, 0.60,0.65,0.70]



class VOC2012(Dataset):
    def __init__(self, voc_directory, train_val=True, difficult=False, transform=False):
        '''
        voc_directory: Directory of VOC, e.g. 'VOCdevkit/VOC2012'
        train_val: True for trainset, False for valset
        difficult: True if including difficult sets
        transform: Transformations
        '''
        self.train_val_directory = voc_directory + '/ImageSets/Main'
        self.image_directory = voc_directory + '/JPEGImages'
        self.train_val = train_val
        self.difficult = difficult
        self.transform = transform
        assert self.train_val == True or self.train_val == False
        files_in_directory = os.listdir(self.train_val_directory)
        assert 'train.txt' in files_in_directory and 'val.txt' in files_in_directory
        if self.train_val:
            self.train_files = [x for x in files_in_directory if x.endswith('_train.txt') and not x.startswith('._')]
            self.train_files.sort()
            num_classes = len(self.train_files)
            self.train_onehot = {}
            self.idx_to_class = {}
            self.idx_to_file = {}
            ctr = 0
            tf = open(self.train_val_directory + '/train.txt', 'r')
            file_idx = 0
            for line in tf:
                self.train_onehot[line.strip()] = [0] * num_classes
                self.idx_to_file[file_idx] = line.strip()
                file_idx += 1
            for filename in self.train_files:
                self.idx_to_class[ctr] = filename[:-10]
                f = open(self.train_val_directory + '/' + filename, 'r')
                for line in f:
                    fname, label = line.split()
                    if label == '0' and self.difficult:
                        self.train_onehot[fname][ctr] = 1
                    if label == '1':
                        self.train_onehot[fname][ctr] = 1
                ctr += 1
            removal = []
            for filename in self.train_onehot:
                if sum(self.train_onehot[filename]) == 0:
                    removal.append(filename)
            for filename in removal:
                del(self.train_onehot[filename])
                        
        else:
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
        if self.train_val:
            return len(self.train_onehot) 
        else:
            return len(self.val_onehot)
    
    def __getitem__(self, idx):
        if self.train_val:
            filename = self.image_directory + '/' + self.idx_to_file[idx] + '.jpg'
            img = Image.open(filename)
            if self.transform:
                img = self.transform(img)
            target = torch.tensor(self.train_onehot[self.idx_to_file[idx]])
            return img, target.float()
        else:
            filename = self.image_directory + '/' + self.idx_to_file[idx] + '.jpg'
            img = Image.open(filename)
            if self.transform:
                img = self.transform(img)
            target = torch.tensor(self.val_onehot[self.idx_to_file[idx]])
            return img, target.float()


def train(model, device, train_loader, optimizer, epochs, log_interval):
    model.train()
    total_loss = 0
    for idx, (data, target) in enumerate(train_loader):
        num_crop = data.shape[1]
        input_size = data.shape[0]
        data = data.view([-1,data.shape[-3],data.shape[-2],data.shape[-1]])
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.view(input_size, num_crop, -1)
        output = output.mean(1)
        output = torch.sigmoid(output)
        loss = F.binary_cross_entropy(output, target)
        total_loss += loss
        loss.backward()
        optimizer.step()
        if idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epochs, idx*train_loader.batch_size, len(train_loader.dataset),
                                                                           100*idx*train_loader.batch_size/len(train_loader.dataset), loss.item()))
    print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epochs, len(train_loader.dataset), len(train_loader.dataset),
                                                                           100*len(train_loader.dataset)/len(train_loader.dataset), loss.item()))
    return total_loss/len(train_loader.dataset)
        

def test(model, device, test_loader, threshold):
    model.eval()
    test_loss = 0
    byclass_accuracy = {}
    total_correct = 0
    total_num = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            num_crop = data.shape[1]
            input_size = data.shape[0]
            data = data.view([-1,data.shape[-3],data.shape[-2],data.shape[-1]])
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = output.view(input_size, num_crop, -1)
            output = output.mean(1)
            output = torch.sigmoid(output)
            test_loss += F.binary_cross_entropy(output, target, reduction='sum').item()
            if type(threshold) == list:
                for thrd in threshold:
                    pred = torch.from_numpy(np.where(output.detach().cpu().numpy()>thrd,1,0))
                    pred = pred.float()
                    for i in range(len(pred)):
                        for j in range(len(pred[i])):
                            total_num += 1
                            if int(pred[i][j]) == 1 and int(target[i][j]) == 1:
                                # TP
                                total_correct += 1
                                if thrd in byclass_accuracy:
                                    if j in byclass_accuracy[thrd]:
                                        if 'TP' in  byclass_accuracy[thrd][j]:
                                            byclass_accuracy[thrd][j]['TP'] += 1
                                        else:
                                            byclass_accuracy[thrd][j]['TP'] = 1
                                    else:
                                        byclass_accuracy[thrd][j] = {'TP':1}
                                else:
                                    byclass_accuracy[thrd] = {j:{'TP':1}}
                            elif int(pred[i][j]) == 0 and int(target[i][j]) == 0:
                                # TN
                                total_correct += 1
                                if thrd in byclass_accuracy:
                                    if j in byclass_accuracy[thrd]:
                                        if 'TN' in  byclass_accuracy[thrd][j]:
                                            byclass_accuracy[thrd][j]['TN'] += 1
                                        else:
                                            byclass_accuracy[thrd][j]['TN'] = 1
                                    else:
                                        byclass_accuracy[thrd][j] = {'TN':1}
                                else:
                                    byclass_accuracy[thrd] = {j:{'TN':1}}
                            elif int(pred[i][j]) == 1 and int(target[i][j]) == 0:
                                # FP
                                if thrd in byclass_accuracy:
                                    if j in byclass_accuracy[thrd]:
                                        if 'FP' in  byclass_accuracy[thrd][j]:
                                            byclass_accuracy[thrd][j]['FP'] += 1
                                        else:
                                            byclass_accuracy[thrd][j]['FP'] = 1
                                    else:
                                        byclass_accuracy[thrd][j] = {'FP':1}
                                else:
                                    byclass_accuracy[thrd] = {j:{'FP':1}}
                            elif int(pred[i][j]) == 0 and int(target[i][j]) == 1:
                                # FN
                                if thrd in byclass_accuracy:
                                    if j in byclass_accuracy[thrd]:
                                        if 'FN' in  byclass_accuracy[thrd][j]:
                                            byclass_accuracy[thrd][j]['FN'] += 1
                                        else:
                                            byclass_accuracy[thrd][j]['FN'] = 1
                                    else:
                                        byclass_accuracy[thrd][j] = {'FN':1}
                                else:
                                    byclass_accuracy[thrd] = {j:{'FN':1}}
            else:
                pred = torch.from_numpy(np.where(output.detach().cpu().numpy()>threshold,1,0))
                pred = pred.float()
                for i in range(len(pred)):
                    for j in range(len(pred[i])):
                        total_num += 1
                        if int(pred[i][j]) == 1 and int(target[i][j]) == 1:
                            # TP
                            total_correct += 1
                            if j in byclass_accuracy:
                                if 'TP' in  byclass_accuracy[j]:
                                    byclass_accuracy[j]['TP'] += 1
                                else:
                                    byclass_accuracy[j]['TP'] = 1
                            else:
                                byclass_accuracy[j] = {'TP':1}
                        elif int(pred[i][j]) == 0 and int(target[i][j]) == 0:
                            # TN
                            total_correct += 1
                            if j in byclass_accuracy:
                                if 'TN' in  byclass_accuracy[j]:
                                    byclass_accuracy[j]['TN'] += 1
                                else:
                                    byclass_accuracy[j]['TN'] = 1
                            else:
                                byclass_accuracy[j] = {'TN':1}
                        elif int(pred[i][j]) == 1 and int(target[i][j]) == 0:
                            # FP
                            if j in byclass_accuracy:
                                if 'FP' in  byclass_accuracy[j]:
                                    byclass_accuracy[j]['FP'] += 1
                                else:
                                    byclass_accuracy[j]['FP'] = 1
                            else:
                                byclass_accuracy[j] = {'FP':1}
                        elif int(pred[i][j]) == 0 and int(target[i][j]) == 1:
                            # FN
                            if j in byclass_accuracy:
                                if 'FN' in  byclass_accuracy[j]:
                                    byclass_accuracy[j]['FN'] += 1
                                else:
                                    byclass_accuracy[j]['FN'] = 1
                            else:
                                byclass_accuracy[j] = {'FN':1}
    
    accuracy = 100.*total_correct/total_num
    test_loss /= len(test_loader.dataset)
    print('\nPerformance: Accuracy: {}/{} ({:.2f}%), Loss: {:.6f}\n'.format(total_correct, total_num, accuracy, test_loss))
    return test_loss, accuracy, byclass_accuracy


fivecrop_traintime = transforms.Compose([
        transforms.Resize(280),
        transforms.ColorJitter(brightness= 2, contrast = 2, hue= 0.5, saturation=2),
        transforms.RandomRotation(20, resample=Image.BILINEAR),
        transforms.FiveCrop(224), 
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda norms: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(norm) for norm in norms]))
    ])

fivecrop_testtime = transforms.Compose([
        transforms.Resize(280),
        transforms.FiveCrop(224), 
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda norms: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(norm) for norm in norms]))
    ])



VOCTrain = VOC2012(VOC_DIRECTORY, train_val = True, transform = fivecrop_traintime)
VOCVal = VOC2012(VOC_DIRECTORY, train_val = False, transform = fivecrop_testtime)
train_loader = DataLoader(VOCTrain, batch_size=4,
                        shuffle=True, num_workers=0) 
val_loader = DataLoader(VOCVal, batch_size=4,
                        shuffle=False, num_workers=0)



resnet = models.resnet18(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_ftrs, 20)
# Continue training on previous model
# resnet.load_state_dict(torch.load('coloradjust/bestresnetweights.pt'), strict=False)

resnet.to(device)
optimizer = optim.SGD(resnet.parameters(), lr=learning_rate, momentum=momentum)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


x_epoch = []
train_loss_ot = []
val_loss_ot = []
val_acc_ot = []
best_loss = sys.maxsize
for epoch in range(start_epoch, start_epoch + epochs):
    train_loss = train(resnet, device, train_loader, optimizer, epoch, log_freq)
    x_epoch.append(epoch)
    train_loss_ot.append(train_loss)
    
    val_loss, val_accuracy, val_byclass_accuracy = test(resnet, device, val_loader, threshold)
    with open('{}/byclassepoch{}.json'.format(OUTPUT_DIRECTORY, epoch), 'w') as fp:
        json.dump(val_byclass_accuracy, fp)
    val_loss_ot.append(val_loss)
    val_acc_ot.append(val_accuracy)
    if val_loss < best_loss:
        torch.save(resnet.state_dict(), '{}/bestresnetweights.pt'.format(OUTPUT_DIRECTORY))
        best_lostt = val_loss

plottingvalues = {'epoch': x_epoch, 'train_loss':train_loss_ot, 'val_loss':val_loss_ot, 'val_acc':val_acc_ot}
plottingvalues['train_loss'] = [float(x) for x in plottingvalues['train_loss']]
with open('{}/resnetplotting.json'.format(OUTPUT_DIRECTORY), 'w') as plotf:
    json.dump(plottingvalues, plotf)


