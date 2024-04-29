from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')

from dataset import yoloCOCO,yoloPascal
from models import Yolodet
from config import dark53_yolo
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.trainer import YolodetTrainer
from utils.checkpoint import load_checkpoint
cfg=dark53_yolo

if cfg.DATASET.NAME == 'coco':
    dataset = yoloCOCO
elif cfg.DATASET.NAME == 'pascal':
    dataset = yoloPascal
else:
    raise Exception("not support")

model = Yolodet(dark53_yolo,pretrained=True)
#load_checkpoint(model,'weights/dark_yolo/old.pth')
train_loader = DataLoader(dataset(cfg, 'train'), batch_size=cfg.DATASET.BATCHSIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset(cfg, 'val'), batch_size=cfg.DATASET.BATCHSIZE, shuffle=True, num_workers=4)
opti = Adam(model.parameters(), lr=cfg.OPTIM.INIT_LR,weight_decay=cfg.OPTIM.WEIGHT_DECAY)
trainer = YolodetTrainer(cfg, model, opti, [train_loader, val_loader])
trainer.set_device('cuda')
trainer.run()