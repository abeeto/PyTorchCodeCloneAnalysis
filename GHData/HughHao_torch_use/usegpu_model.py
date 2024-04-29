# -*- coding: utf-8 -*-
# @Time : 2022/4/12 22:15
# @Author : hhq
# @File : usegpu_model.py
import torch as t
use_gpu = t.cuda.is_available()  # gpu使用
# todo 1.构建网络时，把网络，与损失函数转换到GPU上
# model = get_model()
# loss_f = t.nn.CrossEntropyLoss()
# if use_gpu:
#     model = model.cuda()
#     loss_f = loss_f.cuda()

# todo 2.训练网络时，把数据转换到GPU上
# if use_gpu:
#     x,y = x.cuda(),y.cuda()

# todo 3.取出数据时，需要从GPU准换到CPU上进行操作
#
# if(use_gpu):
#     loss = loss.cpu()
#     acc = acc.cpu()

# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model=model.to(device)
# x=x.to(device)
# y=y.to(device)