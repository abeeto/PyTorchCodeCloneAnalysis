# -*- encoding: utf-8 -*-
"""
@File    :   data.py.py    
@Contact :   thgpddl@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/29 19:52   thgpddl      1.0         None
"""
from torchvision.datasets import MNIST
import torchvision.transforms as transforms  # 用于图像变换
from torch.utils.data import DataLoader

def getLoader():
    # ToTensor()：Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
    # 也即是该函数同时将图片标准化到[0,1]了
    transform=transforms.Compose([transforms.Resize((32,32)),
                                  transforms.ToTensor()])
    data_train=MNIST("./data",download=True,transform=transform)  # 下载到data文件夹

    data_test=MNIST("./data",train=False,download=True,transform=transform)

    # 输出256*1*32*32大小的张量
    data_train_loader=DataLoader(data_train,batch_size=256,shuffle=True)
    data_test_loader=DataLoader(data_test,batch_size=1024)
    return data_train_loader,data_test_loader



