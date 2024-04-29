import os

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import alexnet, vgg16
from torchvision.ops import roi_pool
from torchsummary import summary

from utils import BASE_DIR


class WSDDN(nn.Module):
    def __init__(self, base_net="vgg"):
        super().__init__()

        assert base_net in {"alexnet", "vgg"}, "`base_net` should be in {alexnet, vgg}"

        self.base_net = base_net

        if self.base_net == "alexnet":
            self.base = alexnet(pretrained=False)
            self.roi_output_size = (6, 6)
        else:
            self.base = vgg16(pretrained=False)
            self.roi_output_size = (7, 7)


        self.features = self.base.features[:-1]
        self.fcs = self.base.classifier[:-1]    #论文中的fc6和fc7
        self.fc_c = nn.Linear(4096, 20)
        self.fc_d = nn.Linear(4096, 20)

    def forward(self, batch_imgs, batch_boxes, batch_scores):
        # assume batch size is 1
        batch_boxes = [batch_boxes[0]]

        out = self.features(batch_imgs)  # [1, 256, 21, 29]


        #以(3, 480, 480)输入为例，经过该层前 shape为（512, 30, 30）
        #即图片缩小到输入的1/16， 相应的框的坐标也缩小到原来的1/16
        out = roi_pool(out, batch_boxes, self.roi_output_size, 1.0 / 16)


        out = out.view(len(batch_boxes[0]), -1) #行数为框数，列数未知（用-1来自动调整）

        out = out * batch_scores[0]  # 每行代表一个框，乘上这个框对应的EB Score
        out = self.fcs(out)  # [4000, 4096]

        classification_scores = F.softmax(self.fc_c(out), dim=1)
        detection_scores = F.softmax(self.fc_d(out), dim=0)
        combined_scores = classification_scores * detection_scores
        return combined_scores

    @staticmethod
    def calculate_loss(combined_scores, target):
        image_level_scores = torch.sum(combined_scores, dim=0)  #对每个区域的得分加和，得到图片总得分(在20个类上)
        image_level_scores = torch.clamp(image_level_scores, min=0.0, max=1.0) #得分小于0的设置为0, 得分在0和1之间的保持不变，得分超过1的为1
        loss = F.binary_cross_entropy(image_level_scores, target, reduction="sum")
        return loss
