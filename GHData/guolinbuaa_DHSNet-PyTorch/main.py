import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
from dataset import MyData, MyTestData
from model import Feature
from model import RCL_Module
import time
import visdom
import os
import numpy as np
from matplotlib.pyplot import imsave
vis = visdom.Visdom()
win0 = vis.image(torch.zeros(3, 224, 224))

train_root = '/home/gwl/datasets/DUT-OMRON/DATASET'  # training dataset
val_root = '/home/gwl/datasets/DUT-OMRON/val'  # validation dataset
check_root = './parameters'  # save checkpoint parameters
val_output_root = './validation'  # save validation results
bsize = 1  # batch size
iter_num = 30  # training epochs


std = [.229, .224, .225]
mean = [.485, .456, .406]

os.system('rm -rf ./runs/*')


if not os.path.exists('./runs'):
    os.mkdir('./runs')

if not os.path.exists(check_root):
    os.mkdir(check_root)

if not os.path.exists(val_output_root):
    os.mkdir(val_output_root)

# models
feature = Feature(RCL_Module)
feature.cuda()
train_loader = torch.utils.data.DataLoader(
    MyData(train_root, transform=True),
    batch_size=bsize, shuffle=True, num_workers=4, pin_memory=True)
#return image,gt

val_loader = torch.utils.data.DataLoader(
    MyTestData(val_root, transform=True),
    batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

criterion = nn.BCEWithLogitsLoss()
optimizer_feature = torch.optim.Adam(feature.parameters(), lr=1e-4)
istep = 0
train = 1
losses = []
feature.load_state_dict(torch.load('feature-epoch-29-step-3000.pth'))
if (train):
    start = time.time()
    for it in range(iter_num):
        for ib, (data, gt) in enumerate(train_loader):
            inputs = Variable(data).cuda()
            gt = Variable(gt.unsqueeze(1)).cuda()
            gt_28 = functional.upsample(gt,size=28,mode='bilinear')
            gt_56 = functional.upsample(gt,size=56,mode='bilinear')
            gt_112 = functional.upsample(gt,size=112,mode='bilinear')

            msk1,msk2,msk3,msk4,msk5 = feature.forward(inputs)
            loss = criterion(msk1, gt_28)+criterion(msk2, gt_28)+criterion(msk3, gt_56)+criterion(msk4, gt_112)+criterion(msk5, gt)
            feature.zero_grad()
            loss.backward()
            optimizer_feature.step()

            losses.append(loss.data[0])
            xl = np.arange(len(losses))
            vis.line(np.array(losses), xl, env='DHS', win=win0, opts=dict(title='loss'))

            print('loss: %.4f (epoch: %d, step: %d)' % (loss.data[0], it, ib))
            if ib % 1000 == 0:
                filename = ('%s/feature-epoch-%d-step-%d.pth' % (check_root, it, ib))
                torch.save(feature.state_dict(), filename)
                print('save: (epoch: %d, step: %d)' % (it, ib))
    end = time.time()
    print(end-start)

else:
    if not os.path.exists(val_output_root):
        os.mkdir(val_output_root)
    for ib, (data, img_name, img_size) in enumerate(val_loader):
        print(ib)

        inputs = Variable(data).cuda()
        _, _, _, _, output = feature.forward(inputs)
        print(output)
        output = functional.sigmoid(output)

        mask = output.data[0,0].cpu().numpy()
        imsave(os.path.join(val_output_root, img_name[0] + '.png'), mask, cmap='gray')


