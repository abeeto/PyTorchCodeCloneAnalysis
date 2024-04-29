# ====================================================
# Library
# ====================================================
from utils import helper as hl
from utils import dataset as dt
from utils import config as cfg
import warnings
import timm
from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.nn.parameter import Parameter
import torchvision.models as models
from torch.optim import Adam, SGD
import torch.nn.functional as F
import torch.nn as nn
import torch
from PIL import Image
import cv2
from functools import partial
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
import numpy as np
import scipy as sp
from collections import defaultdict, Counter
from contextlib import contextmanager
from pathlib import Path
import shutil
import random
import time
import math
# import os
import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')

from utils import config as cfg
from utils import dataset as dt
from utils import helper as hl
from train  import train_loop
warnings.filterwarnings('ignore')

LOGGER = hl.init_logger()

if cfg.CFG.apex:
    from apex import amp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hl.seed_torch(seed=cfg.CFG.seed)

DATA_PATH = '../input/cassava-leaf-disease-classification/'

TRAIN_PATH = DATA_PATH + 'train_images'
TEST_PATH = DATA_PATH + 'test_images'
OUTPUT_DIR = './'


# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):

    if data == 'train':
        return Compose([
            #Resize(CFG.size, CFG.size),
            RandomResizedCrop(cfg.CFG.size, cfg.CFG.size),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            ShiftScaleRotate(
                p=0.5,
                shift_limit=(-0.3, 0.3),
                scale_limit=(-0.1, 0.1),
                rotate_limit=(-180, 180),
                interpolation=0,
                border_mode=4,),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(cfg.CFG.size, cfg.CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


# ====================================================
# MODEL
# ====================================================
class CustomEfficient(nn.Module):
    def __init__(self, model_name='efficientnet_b3', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, cfg.CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x


# ====================================================
# main
# ====================================================
def main():
    """
    Prepare: 1.train  2.test  3.submission  4.folds
    """
    train = pd.read_csv(DATA_PATH + 'train.csv')
    test = pd.read_csv(
        DATA_PATH + 'sample_submission.csv')
    # label_map = pd.read_json(DATA_PATH + 'label_num_to_disease_map.json',
    #                         orient='index')

    folds = train.copy()
    Fold = StratifiedKFold(n_splits=cfg.CFG.n_fold,
                       shuffle=True, random_state=cfg.CFG.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds[cfg.CFG.target_col])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    model = CustomEfficient(model_name=cfg.CFG.model_name, pretrained=False)
    train_dataset = dt.TrainDataset(train, transform=get_transforms(data='train'))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True)



    def get_result(result_df):
        preds = result_df['preds'].values
        labels = result_df[cfg.CFG.target_col].values
        score = hl.get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.5f}')

    if cfg.CFG.train:
        # train
        oof_df = pd.DataFrame()
        for fold in range(cfg.CFG.n_fold):
            if fold in cfg.CFG.trn_fold:
                _oof_df = train_loop(folds, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        # CV result
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        # save result
        oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)

    if cfg.CFG.inference:
        # inference
        model = CustomEfficient(cfg.CFG.model_name, pretrained=False)
        states = [torch.load(
            OUTPUT_DIR+f'{cfg.CFG.model_name}_fold{fold}_best.pth') for fold in CFG.trn_fold]
        test_dataset = dt.TestDataset(
            test, transform=get_transforms(data='valid'))
        test_loader = DataLoader(test_dataset, batch_size=cfg.CFG.batch_size, shuffle=False,
                                 num_workers=cfg.CFG.num_workers, pin_memory=True)
        predictions = dt.inference(model, states, test_loader, device)
        # submission
        test['label'] = predictions.argmax(1)
        test[['image_id', 'label']].to_csv(
            OUTPUT_DIR+'submission.csv', index=False)

if __name__ == "__main__":
    main()
