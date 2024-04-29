#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 14:06:25 2018

@author: jangqh
"""
import pandas as pd
import torch
from torch.autograd import Variable

"""
1、DataSet 数据集
"""
class myDataset(Dataset):
    def __init__(self, csv_file, txt_file, root_dir, other_file):
        self.csv_data = pd.read_csv(csv_file)
        with open(txt_file, 'r') as f:
            data_list = f.readlines()
        self.txt_data = data_list
        self.root_dir = root_dir
    
    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, idx):
        data = (self.csv_data[idx], self.txt_data[idx])
        return data
    
dataiter = DataLoader(myDataset, batch_size = 32, shuffle = True, collate_fn_collate)

"""
2、Module 模组
"""
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)

"""
3、optim 优化
"""
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)


"""
4、模型的保存和加载
"""
torch.save(model, './model.pth')  #保存整个模型的结构和参数
torch.save(model.state_dict(), './model_state.pth')   #保存模型的参数

load_model = torch.load('model.pth')  #加载模型的结构和参数

model.load_state_dic(torch.load('model_state.pth'))
























