import torch
import torch.nn as nn
import math


class dehaze_net(nn.Module):

    def __init__(self):
        super(dehaze_net, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(12, 2, 3, 1, 1, bias=True)

    # willy test
    # self.e_conv5 = nn.Conv2d(12,3,3,1,1,bias=True)

    def forward(self, x):
        source = [x]

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))

        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.e_conv3(concat1))

        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))

        concat3 = torch.cat((x1, x2, x3, x4), 1)

        # TODO: (Finish): Way 1: e_conv5 (12:2) 當作 UV

        # 1. x 取sub matrix (沒有y)
        x5 = self.relu(self.e_conv5(concat3))
        x_sub = torch.narrow(x, 1, 1, 2)
        clean_image = self.relu((x5 * x_sub) - x5 + 1)

        # 2. 完成 clean_image 後把y補回去
        x_sub = torch.narrow(x, 1, 0, 1)
        clean_image = torch.cat((x_sub, clean_image), 1)

        # TODO: Wat 2: 原本 e_conv5 (12:3) , 再裁掉 Y
        '''
        x5 = self.relu(self.e_conv5(concat3))
        clean_image = self.relu((x5 * x_sub) - x5 + 1)
        clean_image = torch.narrow(clean_image,)
        '''

        # clean_image = self.relu((x5 * x) - x5 + 1)

        return clean_image
