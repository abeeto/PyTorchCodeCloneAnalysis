from __future__ import division

from utils.datasets import *

import torch
from torch.utils.data import DataLoader

train_path='data/coco_my_sample/trainvalno5k.txt'

dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=2, shuffle=False, num_workers=0
)

for batch_i, (_, imgs, targets) in enumerate(dataloader):
    print(_)
    print(targets.shape)
    break