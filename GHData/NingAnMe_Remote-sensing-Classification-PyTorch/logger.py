#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020-12-14 17:24
# @Author  : NingAnMe <ninganme@qq.com>

from flyai.utils.log_helper import train_log as tl


def train_log(loss_train=0.0, acc_train=0.0, loss_val=0.0, acc_val=0.0):
    loss_train = loss_train if loss_train is not None else 0.0
    acc_train = acc_train if acc_train is not None else 0.0
    loss_val = loss_val if loss_val is not None else 0.0
    acc_val = acc_val if acc_val is not None else 0.0
    tl(train_loss=float('{:0.4f}'.format(loss_train)),
       train_acc=float('{:0.4f}'.format(acc_train)),
       val_loss=float('{:0.4f}'.format(loss_val)),
       val_acc=float('{:0.4f}'.format(acc_val)))
