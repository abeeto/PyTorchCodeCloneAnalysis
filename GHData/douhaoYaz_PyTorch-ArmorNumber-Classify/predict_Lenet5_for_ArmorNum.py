# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 20:33:10 2021

@author: 逗号@东莞理工ACE实验室
"""
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torchvision as tv
import torchvision.transforms as transforms
import torch
from PIL import Image
import torch.nn as nn
from Lenet5 import Lenet5

input_size = 48
names = ['0', '1', '2', '3', '4', '5']
def pridict():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=Lenet5()
    model.load_state_dict(torch.load("Lenet5_v2.pth"))
    model = model.to(device)
    model.eval()  # 推理模式

    # 获取测试图片，并行相应的处理
    # img = Image.open('test.jpg')
    img = Image.open('test.png')
    # 查看转换前的img的格式
    print("img shape before trransform:", img)
    transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    img = transform(img)
    # 查看转换后的img的格式
    print('img shape after transform:', img.shape)
    # print('img pixel after transform:', img)
    img = img.unsqueeze(0)
    # 查看unsqueeze之后的img的格式
    print('img shape after unsqueeze:', img.shape)
    img = img.to(device)


    with torch.no_grad():
        py = model(img)
    _, predicted = torch.max(py, 1)  # 获取分类结果
    classIndex_ = predicted[0]

    print('predict:', py)
    print('预测结果：', names[classIndex_])


if __name__ == '__main__':
    pridict()
