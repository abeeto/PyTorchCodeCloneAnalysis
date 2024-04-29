#!/usr/bin/env python
##########################################################
# File Name: eval.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-05-11 09:37:39
##########################################################

import torch 
import torchvision.transforms as transforms

from model.simple_resnet import Generator 
from illu_data.illu_dataset import IlluDataset
import torchvision.utils as vutils
#Generator.load_state_dict(

#best_model = "zoo/MSE/netG_epoch_55.pth"
best_model = "snapshots/netG_epoch_20.pth"

dataset = IlluDataset(root="/data3/lzh/10000x672x672_box_diff/",
                transform=transforms.Compose([
                    transforms.ToTensor(),
                ]),
                target_transform = transforms.Compose([
                    transforms.ToTensor(),
                ]))
dataset.out_size = (672, 672)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=1)

device = torch.device("cuda:0")
netG = Generator(1, [3,3,3], 30).to(device)
netG.load_state_dict(torch.load(best_model))

out = "benchmark/"

for i, data_label in enumerate(dataloader): 
    if i >= 10:
        break
    data, groundtruth = data_label
    real_cpu = groundtruth.to(device)
    noise = data.to(device)
    fake = netG(noise)
    vutils.save_image(real_cpu,
            '%s/real_samples.png' % out,
            normalize=True)
    #fake = netG(fixed_noise)
    vutils.save_image(fake.detach().clamp(0.0, 1.0),
            '%s/fake_samples_epoch_%03d.png' % (out, i),
            normalize=True)
    vutils.save_image(torch.cat([fake.detach().clamp(0.0, 1.0), real_cpu], dim = 3),
            '%s/gan_res%d.png' % (out, i),
            normalize=True)

