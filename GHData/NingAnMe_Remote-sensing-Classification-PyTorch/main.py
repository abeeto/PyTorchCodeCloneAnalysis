#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020-12-13 15:07
# @Author  : NingAnMe <ninganme@qq.com>
import os
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from flyai.dataset import Dataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import ClassifierDataset
from net import get_net
from loss import LabelSmoothCrossEntropyLoss
from optimizer import Ranger
from path import DATA_ROOT_PATH, MODEL_PATH
from logger import train_log


############
# 辅助函数
############


# def f1_score(preds, target):
#     preds = torch.argmax(preds, dim=1)
#     return plm.f1(preds, target, num_classes=Num_classes)


class Classifier(pl.LightningModule):
    def __init__(self, net_name, train_layer="None",
                 loss_function='CrossEntropyLoss',
                 optim='Adam', lr=1e-3,
                 local=False, ):
        """
        :param net_name (str):  网络名称
        :param train_layer (str): 从某一层开始恢复训练
        :param loss_function (str):  损失函数名称
        :param optim (str):  优化器名称
        :param lr (float):  学习率
        :param local (bool):  是否本地训练
        """
        super(Classifier, self).__init__()
        self.m_net = get_net(net_name, train_layer=train_layer)
        self.m_loss_function = self.configure_loss_function(loss_function)
        self.m_optim = optim
        self.m_lr = lr
        self.m_accuracy_train = pl.metrics.Accuracy()
        self.m_accuracy_val = pl.metrics.Accuracy()

        self.local = local

        self.l_loss_train = None
        self.l_loss_val = None
        self.l_acc_train = None
        self.l_acc_val = None

    def forward(self, x):
        return self.m_net(x)

    @staticmethod
    def configure_loss_function(loss_function):
        # 损失函数
        if loss_function == 'CrossEntropyLoss':
            lf = torch.nn.CrossEntropyLoss()
        elif loss_function == 'LabelSmoothCrossEntropyLoss':
            lf = LabelSmoothCrossEntropyLoss(smoothing=0.1)
        else:
            raise ValueError(loss_function)
        return lf

    def configure_optimizers(self):
        if self.m_optim == 'Adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.m_lr)
        elif self.m_optim == 'Ranger':
            optim = Ranger(self.parameters(), lr=self.m_lr)
        else:
            raise ValueError(self.m_optim)
        MAX_STEP = int(1e4)
        # 继续训练的时候，需要基于恢复的optimizer重定义lr_scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, MAX_STEP, eta_min=1e-5)
        return {
            'optimizer': optim,
            'lr_scheduler': lr_scheduler,
            # 'monitor': 'metric_to_track',
            'interval': 'step',
            'frequency': 1,
            'strict': True,
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.m_net(x)
        loss = self.m_loss_function(pred, y)
        self.m_accuracy_train.update(pred, y)
        acc = self.m_accuracy_train.compute()

        train_log(loss_train=loss,
                  acc_train=acc,
                  loss_val=self.l_loss_val,
                  acc_val=self.l_acc_val)
        return loss

    def training_epoch_end(self, outputs):
        nums = len(outputs)
        self.l_loss_train = None
        for i in outputs:
            self.l_loss_train = self.l_loss_train + i['loss'] if self.l_loss_train is not None else i['loss']
        self.l_loss_train /= nums
        self.l_acc_train = self.m_accuracy_train.compute()
        self.m_accuracy_train.reset()

        self.log('loss_train', self.l_loss_train)
        self.log('acc_train', self.l_acc_train)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.m_net(x)
        loss = self.m_loss_function(pred, y)
        self.m_accuracy_val.update(pred, y)

        return loss

    def validation_epoch_end(self, outputs):
        nums = len(outputs)
        self.l_loss_val = None
        for i in outputs:
            self.l_loss_val = self.l_loss_val + i if self.l_loss_val is not None else i
        self.l_loss_val /= nums

        self.l_acc_val = self.m_accuracy_val.compute()
        self.m_accuracy_val.reset()

        self.log('loss_val', self.l_loss_val)
        self.log('acc_val', self.l_acc_val)

        train_log(loss_train=self.l_loss_train,
                  acc_train=self.l_acc_train,
                  loss_val=self.l_loss_val,
                  acc_val=self.l_acc_val)

    def test_step(self, batch, batch_idx):
        """
        model.eval() and torch.no_grad() are called automatically for testing.
        The test loop will not be used until you call: trainer.test()
        .test() loads the best checkpoint automatically
        """
        x, y = batch
        pred = self.forward(x)
        loss = self.m_loss_function(pred, y)
        return loss


class FlyAiCallback(Callback):

    def __init__(self):
        super().__init__()

    def on_save_checkpoint(self, trainer, pl_module):
        print('epoch_{}_step_{}'.format(trainer.current_epoch, trainer.global_step))
        dir_path = os.path.dirname(MODEL_PATH)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        torch.save(pl_module.m_net, MODEL_PATH)


def main():
    """
    项目的超参
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=1, type=int, help="batch size")
    args = parser.parse_args()

    local = True  # True  False
    if local:
        train_layer = 'fc'
        batch_size = 4
        max_epoch = 50
        es_patience = 3
        n_workers = 0
    else:
        # train_layer = '_blocks.46'
        train_layer = 'conv'
        batch_size = 16
        max_epoch = 10
        es_patience = 3
        n_workers = 0

    net_name = 'efficientnet-b4'  # resnet50  resnext101_32x8d  efficientnet-b7  efficientnet-b4
    loss_name = 'LabelSmoothCrossEntropyLoss'  # CrossEntropyLoss  LabelSmoothCrossEntropyLoss
    optim_name = 'Adam'  # Adam  Ranger
    lr_scheduler_name = None

    # 数据
    dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
    images_train, labels_train, images_val, labels_val = dataset.get_all_data()
    print('len images_train: {}'.format(len(images_train)))
    print('len images_val  : {}'.format(len(images_val)))

    dataset_train = ClassifierDataset(images_train, labels_train)
    dataset_val = ClassifierDataset(images_val, labels_val)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, num_workers=n_workers)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, num_workers=n_workers)
    print('len train_loader: {}'.format(len(train_loader)))
    print('len val_loader: {}'.format(len(val_loader)))

    model = Classifier(net_name, loss_function=loss_name, optim=optim_name, local=local, train_layer=train_layer)

    trainer = pl.Trainer(gpus=1,
                         default_root_dir=DATA_ROOT_PATH,
                         callbacks=[FlyAiCallback(),
                                    EarlyStopping(monitor='loss_val', patience=es_patience, mode='min')],
                         max_epochs=max_epoch)
    trainer.fit(model, train_loader, val_loader)

    # 最后通过trainer.checkpoint_callback可以获取很多信息,是ModelCheckpoint的实例化
    # from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':

    main()
