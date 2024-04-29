import torch 
import torch.nn as nn

'''
引用 'Focal Loss for Dense Object Detection' 论文原文：
'As in [20] all pyramid levels have C = 256 channels.'
因此，需要将resnet的多尺度特征图通道数通过卷积层转换为256
其中，resnet输出的多尺度特征图大小分别为：
输入图像：torch.Size([1, 3, 640, 640]) 的情况下
C3: torch.Size([1, 512, 80, 80])
C4: torch.Size([1, 1024, 40, 40])
C5: torch.Size([1, 2048, 20, 20])
'''

class FPN(nn.Module):
    def __init__(self, C3_channel=512, C4_channel=1024, C5_channel=2048, fpn_channel=256):
        super(FPN, self).__init__()
        '''
        引用 'Focal Loss for Dense Object Detection' 论文原文：
        'RetinaNet uses feature pyramid levels P3 to P7, 
        where P3 to P5 are computed from the output of the corresponding ResNet 
        residual stage (C3 through C5) using top-down and lateral connections 
        just as in [20], P6 is obtained via a 3×3 stride-2 conv on C5, 
        and P7 is computed by applying ReLU followed by a 3×3 stride-2 conv on P6.'

        因此，我们需要定义生成P3到P7的网络模型结构
        '''
        # P5
        # 通过1x1卷积核，将C5的通道数降为256，[256,20,20]
        self.P5_conv1 = nn.Conv2d(C5_channel, fpn_channel, kernel_size=1, stride=1, padding=0)
        # 上采样P5_conv1后的特征图，[256,40,40]
        self.P5_up = nn.Upsample(scale_factor=2, mode='nearest')
        # 卷积P5_conv1后的特征图，[256,20,20]
        self.P5_conv2 = nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, stride=1, padding=1)

        # P4
        # 通过1x1卷积核，将C4的通道数降为256，[256,40,40]
        self.P4_conv1 = nn.Conv2d(C4_channel, fpn_channel, kernel_size=1, stride=1, padding=0)
        # 上采样 [P4_conv1后的特征图 + P5_up后的特征图](数值维度)，[256,80,80]
        self.P4_up = nn.Upsample(scale_factor=2, mode='nearest')
        # 卷积 [P4_conv1后的特征图 + P5_up后的特征图](数值维度)，[256,40,40]
        self.P4_conv2 = nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, stride=1, padding=1)

        # P3
        # 通过1x1卷积核，将C3的通道数降为256，[256,80,80]
        self.P3_conv1 = nn.Conv2d(C3_channel, fpn_channel, kernel_size=1, stride=1, padding=0)
        # 卷积 [P3_conv1后的特征图 + P4_up后的特征图](数值维度)，[256,80,80]
        self.P3_conv2 = nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, stride=1, padding=1)

        # P6
        # 通过步长为2的3x3卷积核，下采样C5，且通道数降为256，[256,10,10]
        self.P6_conv = nn.Conv2d(C5_channel, fpn_channel, kernel_size=3, stride=2, padding=1)

        # P7 
        self.P7_relu = nn.ReLU() 
        # 通过步长为2的3x3卷积核，下采样conv_P6输出的特征图，[256,5,5]
        self.P7_conv = nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, stride=2, padding=1)

    def forward(self, C3, C4, C5):
        P5 = self.P5_conv1(C5)
        Up5 = self.P5_up(P5)
        P5 = self.P5_conv2(P5)

        P4 = self.P4_conv1(C4)
        P4 = P4 + Up5
        Up4 = self.P4_up(P4)
        P4 = self.P4_conv2(P4)

        P3 = self.P3_conv1(C3)
        P3 = P3 + Up4
        P3 = self.P3_conv2(P3)

        P6 = self.P6_conv(C5)

        P7 = self.P7_relu(P6)
        P7 = self.P7_conv(P7)

        return P3, P4, P5, P6, P7

if __name__ == '__main__':
    import numpy as np 
    C3 = np.random.uniform(size=[1, 512, 80, 80])
    C4 = np.random.uniform(size=[1, 1024, 40, 40])
    C5 = np.random.uniform(size=[1, 2048, 20, 20])
    C3 = torch.Tensor(C3)
    C4 = torch.Tensor(C4)
    C5 = torch.Tensor(C5)
    print('C3:', C3.size())
    print('C4:', C4.size())
    print('C5:', C5.size())

    fpn = FPN()
    P3, P4, P5, P6, P7 = fpn(C3, C4, C5)
    print('P3:', P3.size())
    print('P4:', P4.size())
    print('P5:', P5.size())
    print('P6:', P6.size())
    print('P7:', P7.size())

'''
C3: torch.Size([1, 512, 80, 80])
C4: torch.Size([1, 1024, 40, 40])
C5: torch.Size([1, 2048, 20, 20])
P3: torch.Size([1, 256, 80, 80])
P4: torch.Size([1, 256, 40, 40])
P5: torch.Size([1, 256, 20, 20])
P6: torch.Size([1, 256, 10, 10])
P7: torch.Size([1, 256, 5, 5])
'''