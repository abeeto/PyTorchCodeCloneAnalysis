'''
test
csv results
'''



import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os

import argparse
import numpy as np 
from torch.optim.lr_scheduler import *
import csv

from net.resnet import resnet101
from Custom_Dataset.DogCat import DogCat 

parser = argparse.ArgumentParser()
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--nepoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--gpu', type=str, default='0,1,2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--pth', type=str, default='../pth/', help='model loaded path')
parser.add_argument('--csvSavepath', type=str, default='../submit/', help='csv results saved path')

opt = parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

testset=DogCat('./data/test',transform=transform_test,train=False,val=False,test=True)
testloader=torch.utils.data.DataLoader(testset,batch_size=opt.batchSize,shuffle=False,num_workers=opt.num_workers)
model=resnet101(pretrained=True)
model.fc=nn.Linear(2048,2)
model.load_state_dict(torch.load(opt.pth+'model.pth'))
model.cuda()
model.eval()
results=[]

with torch.no_grad():
    for image,label in testloader:
        image=image.cuda()
        out=model(image)
        label=label.numpy().tolist()
        _,predicted=torch.max(out.data,1)
        predicted=predicted.data.cpu().numpy().tolist()
        results.extend([[i,";".join(str(j))] for (i,j) in zip(label,predicted)])


test_csv=os.path.join(opt.csvSavepath,'testResult.csv')

with open(test_csv,'w',newline='') as f:
    writer=csv.writer(f,delimiter=',')
    q=("id","label")
    writer.writerow(q)
    for x in results:
        writer.writerow(x)