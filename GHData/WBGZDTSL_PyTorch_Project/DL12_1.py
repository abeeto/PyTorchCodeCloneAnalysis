# 随机模块
import random
# 绘图模块
import matplotlib as mpl
import matplotlib.pyplot as plt
# numpy
import numpy as np
# pytorch
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import Dataset,TensorDataset,DataLoader
#%%
num_inputs = 2               # 两个特征
num_examples = 1000          # 总共一千条数据
torch.manual_seed(420)
#%%
def tensorGenReg(num_examples = 1000, w = [2, -1, 1], bias = True, delta = 0.01, deg = 1):
    """回归类数据集创建函数。

    :param num_examples: 创建数据集的数据量
    :param w: 包括截距的（如果存在）特征系数向量
    :param bias：是否需要截距
    :param delta：扰动项取值
    :param deg：方程次数
    :return: 生成的特征张和标签张量
    """
    if bias == True:
        num_inputs = len(w) - 1  # 特征张量
        features_true = torch.randn(num_examples, num_inputs)  # 不包含全是1的列的特征张量
        w_true = torch.tensor(w[:-1]).reshape(-1, 1).float()  # 自变量系数
        b_true = torch.tensor(w[-1]).float()  # 截距
        if num_inputs == 1:  # 若输入特征只有1个，则不能使用矩阵乘法
            labels_true = torch.pow(features_true, deg) * w_true + b_true
        else:
            labels_true = torch.mm(torch.pow(features_true, deg), w_true) + b_true
        features = torch.cat((features_true, torch.ones(len(features_true), 1)), 1)  # 在特征张量的最后添加一列全是1的列
        labels = labels_true + torch.randn(size=labels_true.shape) * delta

    else:
        num_inputs = len(w)
        features = torch.randn(num_examples, num_inputs)
        w_true = torch.tensor(w).reshape(-1, 1).float()
        if num_inputs == 1:
            labels_true = torch.pow(features, deg) * w_true
        else:
            labels_true = torch.mm(torch.pow(features, deg), w_true)
        labels = labels_true + torch.randn(size=labels_true.shape) * delta
    return features, labels
#%%\
torch.manual_seed(420)
f, l = tensorGenReg(delta=0.01)
#%%
# 设置随机数种子
torch.manual_seed(420)
# 创建初始标记值
num_inputs = 2
num_examples = 500
# 创建自变量簇
data0 = torch.normal(4, 2, size=(num_examples, num_inputs))
data1 = torch.normal(-2, 2, size=(num_examples, num_inputs))
data2 = torch.normal(-6, 2, size=(num_examples, num_inputs))
# 创建标签
label0 = torch.zeros(500)
label1 = torch.ones(500)
label2 = torch.full_like(label1, 2) #2为填充的数值
# 合并生成最终数据
features = torch.cat((data0, data1, data2)).float()
labels = torch.cat((label0, label1, label2)).long().reshape(-1, 1)
#%%
b = np.random.randint(0,2,10)