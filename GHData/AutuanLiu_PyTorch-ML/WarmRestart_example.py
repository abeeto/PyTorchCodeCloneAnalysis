#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Description :  utils.lrs_scheduler WarmRestart and warm_restart example
    Email : autuanliu@163.com
    Date：2018/04/01
"""
from models.utils.utils_imports import *
from models.utils.lrs_scheduler import WarmRestart, warm_restart
from models.vislib.line_plot import line


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


net = Net()
opt = optim.SGD([{'params': net.conv1.parameters()}, {'params': net.conv2.parameters(), 'lr': 0.5}], lr=0.05)

# CosineAnnealingLR with warm_restart
# scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=20, eta_min=0)

scheduler = WarmRestart(opt, T_max=20, T_mult=2, eta_min=1e-10)

vis_data = []
for epoch in range(200):
    scheduler.step()

    # for warm_restart
    # scheduler = warm_restart(scheduler, T_mult=2)

    print(scheduler.get_lr())
    vis_data.append(scheduler.get_lr()[0])
    opt.step()

line(vis_data)
plt.show()
