'''
Description: 
Version: 2.0
Autor: CHEN JIE
Date: 2020-09-21 09:56:58
LastEditors: CHEN JIE
LastEditTime: 2020-09-22 20:25:55
language: 
Deep learning framework: Pytorch1.4.0
'''

import torch
from torch import nn

class Lenet5(nn.Module):
    # for cifar10 dataset
    def __init__(self):
        #调用类的初始化方法来初始化父类
        super(Lenet5, self).__init__()

    #构建网络框架
        self.conv_unit = nn.Sequential(
            # Conv2d(in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t=..., padding: _size_2_t=..., dilation: _size_2_t=..., groups: int=..., bias: bool=..., padding_mode: str=...)
            #原始输入的RGB三通道的图片，所以第一个input_channel==3;
            #kernel_size是卷积核的大小，即一次关注像素点的大小
            #stride是步长
            #padding是填充的大小
            #第一层   x :[b, 3 , 32, 32] => [b, 6, ....]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),#池化层

            #
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            #  之后要更一个全连接层相连，前面的高维必须“打平Flatten”
            # flatten这个类，torch没有；而在Sequential里面必须使用自有的类，所以要搞两个unit
        )
        #flatten
        #fc unit全连接层
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10) 
        )

        

        # # [b, 3, 32, 32]
        # tmp = torch.randn(2, 3, 32, 32)
        # out = self.cov_unit(tmp)
        # # [b, 16, 5, 5]
        # print('conv out:', out.shape)

    def forward(self, x):
        '''
        description: 
        param {type} :x [b, 3, 32, 32]
        return {type} 
        '''            
        batchsz = x.size(0)
        x = self.conv_unit.forward(x)
        x = x.view(batchsz, 16*5*5)
        logits = self.fc_unit(x)

        return logits

def main():
    net = Lennt5()
    # [b, 3, 32, 32]
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    # [b, 16, 5, 5]
    print('lenet out:', out.shape)


if __name__ == "__main__":
    main()
    

