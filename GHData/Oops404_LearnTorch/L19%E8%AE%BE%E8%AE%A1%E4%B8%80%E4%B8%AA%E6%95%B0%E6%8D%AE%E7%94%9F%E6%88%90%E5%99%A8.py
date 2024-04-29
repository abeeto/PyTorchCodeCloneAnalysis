# -*- coding: UTF-8-*-
"""
@Project: LearnTorch
@Author: Oops404
@Email: cheneyjin@outlook.com
@Time: 2022/1/24 14:18
"""
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader

"""
炼丹师(ง •_•)ง的成长之路，生成器的目的是为了控制变量。
"""


# # 输入为2个特征，1000条样本
# num_inputs = 2
# num_examples = 1000
#
# torch.manual_seed(996)
#
# """
# 构建方程 y=2x_1 - x_2 + 1
# """
# # w系数
# coef_w = torch.tensor([2., -1]).reshape(2, 1)
# # b截距
# coef_b = torch.tensor(1.)
#
# features = torch.randn(num_examples, num_inputs)
# # 构建完成
# labels_true = torch.mm(features, coef_w) + coef_b
# # 撒上噪声,randn生成的随机数本身满足正态分布
# labels = labels_true + torch.randn(size=labels_true.shape) * 0.01
#
# # # 绘制一下，121：12表示一行两列，1代表绘制第1个
# # plt.subplot(121)
# # plt.scatter(features[:, 0], labels)
# # # 2代表绘制第2个
# # plt.subplot(122)
# # plt.scatter(features[:, 1], labels)
# # plt.show()
#
# labels1 = labels_true + torch.randn(size=labels_true.shape) * 2
#
# plt.subplot(221)
# plt.scatter(features[:, 0], labels)
# # 2代表绘制第2个
# plt.subplot(222)
# plt.plot(features[:, 1], labels, 'ro')
#
# plt.subplot(223)
# plt.scatter(features[:, 0], labels1)
# # 2代表绘制第2个
# plt.subplot(224)
# plt.plot(features[:, 1], labels1, 'yo')
#
# plt.show()


def data_reg_generator(num_examples=1000, w=None, bias=True, delta=0.01, deg=1):
    """
    综上思路，设计回归类数据生成器:
    :param num_examples: 数据量
    :param w: 自变量系数关系
    :param bias: 截距
    :param delta: 扰动项参数
    :param deg: 方程次数
    :return: 生成特征张量和标签张量
    """
    if w is None:
        w = [2, -1, 1]
    if bias:
        num_inputs = len(w) - 1
        features_true = torch.randn(num_examples, num_inputs)
        w_true = torch.tensor(w[:-1]).reshape(-1, 1).float()
        b_true = torch.tensor(w[-1]).float()
        if num_inputs == 1:
            labels_true = torch.pow(features_true, deg) * w_true + b_true
        else:
            # 避免非常复杂，对于交叉项有不足，if deg!=1
            labels_true = torch.mm(torch.pow(features_true, deg), w_true) + b_true
        features = torch.cat((features_true, torch.ones(len(features_true), 1)), 1)
        labels = labels_true + torch.randn(size=labels_true.shape) * delta
    else:
        num_inputs = len(w)
        features = torch.randn(num_examples, num_inputs)
        w_true = torch.tensor(w).reshape(-1, 1).float()
        if num_inputs == 1:
            labels_true = torch.pow(features, deg) * w_true
        else:
            labels_true = torch.mm(torch.pow(features, deg), w_true)
        labels = labels_true * torch.randn(size=labels_true.shape) * delta
    return features, labels


# torch.manual_seed(996)
# _features, _labels = data_reg_generator(deg=2, delta=0.15)
# plt.subplot(121)
# plt.scatter(_features[:, 0], _labels)
# plt.subplot(122)
# plt.scatter(_features[:, 1], _labels)
# plt.show()

# -------------------------------------------------------------------------
# 多分类数据生成器
#
# 随机生成符合，4均值，2标准差的10行2列数据
# torch.normal(4, 2, size=(10, 2))

# torch.manual_seed(996)
# _num_inputs = 2
# _num_examples = 500
#
# # 自变量簇
# data0 = torch.normal(4, 2, size=(_num_examples, _num_inputs))
# data1 = torch.normal(-2, 2, size=(_num_examples, _num_inputs))
# data2 = torch.normal(-6, 2, size=(_num_examples, _num_inputs))
#
# # 创建012标签
# label0 = torch.zeros(500)
# label1 = torch.ones(500)
# label2 = torch.full_like(label1, 2)
#
# _features = torch.cat((data0, data1, data2)).float()
# _labels = torch.cat((label0, label1, label2)).long().reshape(-1, 1)
#
# plt.scatter(_features[:, 0], _features[:, 1], c=_labels)
# plt.show()

def data_class_generator(num_examples=1000, num_inputs=2, num_class=3, deg_dispersion=None, bias=False):
    """
    分类数据集生成器
    :param num_examples: 每隔类别的数据量
    :param num_inputs: 数据集特征数量
    :param num_class: 分类标签总数
    :param deg_dispersion: 数据分布离散程度参数
    :param bias: 建立模型逻辑回归模型时是否带入截距
    :return: 生成特征张量和标签张量，其中特征张量为二维浮点数组
    """
    if deg_dispersion is None:
        deg_dispersion = [4, 2]

    # 每一类特征张量的均值参考值
    _mean = deg_dispersion[0]
    # 每一类特征张量的方差
    _std = deg_dispersion[1]
    # 用于存储每一类特征张量的列表容器
    lf = []
    # 用于存储每一类特征张量的列表容器
    ll = []
    # 每一类特征张量的惩罚因子
    k = _mean * (num_class - 1) / 2

    for i in range(num_class):
        # 生成每一类的张量
        data_temp = torch.normal(
            i * _mean - k,  # 尽可能围绕0分布
            _std,
            size=(num_examples, num_inputs)
        )
        lf.append(data_temp)
        # 生成每一类的标签，torch.empty(num_examples, 1)每一类标签张量的形状
        labels_temp = torch.full_like(torch.empty(num_examples, 1), i)
        ll.append(labels_temp)

    _features = torch.cat(lf).float()
    _labels = torch.cat(ll).long()

    if bias:
        _features = torch.cat((_features, torch.ones(len(_features), 1)), 1)
    return _features, _labels


_f1, _l1 = data_class_generator(deg_dispersion=[6, 2])
_f2, _l2 = data_class_generator(deg_dispersion=[6, 4])

plt.subplot(121)
plt.scatter(_f1[:, 0], _f1[:, 1], c=_l1)
plt.subplot(122)
plt.scatter(_f2[:, 0], _f2[:, 1], c=_l2)
plt.show()
