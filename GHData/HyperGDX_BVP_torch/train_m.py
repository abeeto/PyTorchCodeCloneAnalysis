import torch
import torch.nn as nn
from model import Widar3_improve

#### config ####
EPOCHS = 100
INIT_LR = 0.001
#### dataset ####
full_dataset =
#### dataloader ####


#### get one batch to show result ####


#### generate & config model ####
model = Widar3_improve(num_classes=6, drop_rate=0.6)
lr_optimizer = torch.optim.Adam(params=model.parameters(), lr=INIT_LR)
# torch.optim.Adam(params=,lr=)
# Adam SGD RMSprop
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(lr_optimizer, milestones=[100, 300, 800], gamma=0.1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 300, 800], gamma=0.1)

#### train validate test ####

### train ###

### validate ###

### test ###

### draw result ###
