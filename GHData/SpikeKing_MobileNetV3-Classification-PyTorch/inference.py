import sys
sys.path.append('./data')
sys.path.append('./model')

import os
import torch
import numpy as np

from torch.nn import functional as F
import torchvision.transforms as transforms
from model import MobileNetV3
from PIL import Image


# 创建一个检测器类，包含了图片的读取，检测等方法
class Detector(object):
    # netkind为'large'或'small'可以选择加载MobileNetV3_large或MobileNetV3_small
    # 需要事先训练好对应网络的权重
    def __init__(self, net_kind, num_classes):
        super(Detector, self).__init__()
        self.num_classes = num_classes
        self.net = MobileNetV3(model_mode=net_kind, num_classes=num_classes,
                               multiplier=1.0, dropout_rate=0.0)
        if torch.cuda.is_available():
            self.net.cuda()
        self.net.eval()

        # filename = "best_model_" + str(net_kind)
        # checkpoint = torch.load('./checkpoint/' + filename + '_ckpt.t7')
        # model.load_state_dict(checkpoint['model'])

    def load_weights(self, weight_path):
        if torch.cuda.is_available():
            checkpoint = torch.load(weight_path)
        else:
            checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))

        new_state_dict = dict()
        for k, v in checkpoint['model'].items():
            name = k[7:]  # remove "module."
            new_state_dict[name] = v

        self.net.load_state_dict(new_state_dict)

    # 检测器主体
    def detect(self, weight_path, pic_path):
        # 先加载权重
        self.load_weights(weight_path=weight_path)

        # self.net = torch.jit.load("checkpoint/region_clz_mv3_best_model_LARGE_ckpt_20220317.ts")
        #
        # # 存储ts模型
        # img = torch.zeros(1, 3, 224, 224).to(torch.device('cuda'))
        # f = weight_path.replace('.t7', '_20220317.ts')  # filename
        # with torch.no_grad():
        #     ts = torch.jit.trace(self.net, img)
        # ts.save(f)
        # print('[Info] 转换ts完成! ')

        # 读取图片
        img = Image.open(pic_path).convert('RGB')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        img_tensor = transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        with torch.no_grad():
            out = self.net(img_tensor)
        probabilities = F.softmax(out[0], dim=0)
        if self.num_classes >= 5:
            top_n = 5
        else:
            top_n = self.num_classes
        top_prob, top_catid = torch.topk(probabilities, top_n)
        top_catid = list(top_catid.cpu().detach().numpy())
        top_prob = list(top_prob.cpu().detach().numpy())
        top_prob = list(np.around(top_prob, 4))
        print('[Info] 预测类别: {}'.format(top_catid))
        print('[Info] 预测概率: {}'.format(top_prob))


if __name__ == '__main__':
    detector = Detector('LARGE', num_classes=3)
    # detector.detect('./mydata/models/best_20210902.pkl', './mydata/document_dataset/000/000002_000.jpg')
    detector.detect('./checkpoint/region_clz_mv3_best_model_LARGE_ckpt_20220317.t7', './mydata/region_clz/0.jpg')
    # detector.detect('./checkpoint/region_clz_mv3_best_model_LARGE_ckpt_20220317.t7', './mydata/region_clz/1.jpg')
    # detector.detect('./checkpoint/region_clz_mv3_best_model_LARGE_ckpt_20220317.t7', './mydata/region_clz/2.jpg')







