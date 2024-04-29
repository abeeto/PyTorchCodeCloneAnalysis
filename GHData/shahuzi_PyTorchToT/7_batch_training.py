#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:     pfwu
@file:       7_batch_training.py
@software:   pycharm
Created on   7/5/18 2:51 PM

"""

import torch as tc
import torch.utils.data as Data

BATCH_SIZE = 5

x = tc.linspace(1,10,10)
y = tc.linspace(10,1,10)


torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

for epoch in range(3):
    for step,(batch_x,batch_y) in enumerate(loader):
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())

