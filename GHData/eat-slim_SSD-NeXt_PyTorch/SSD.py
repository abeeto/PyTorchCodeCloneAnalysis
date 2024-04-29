import torch
import torchvision
from torch import nn
from torch.nn import functional as f
import torch.nn.init as init
from utils.l2norm import L2Norm
from BoundingBox import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vgg16_weights_file = r'weights/vgg16-conv5_3.pth'

vgg16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
extra_layers = [1024, 1024, 256, 512, 128, 256, 128, 256, 128, 256]
bounding_box = [4, 6, 6, 6, 4, 4]
conv4_3 = 12
'''
输入图像为300x300x3，通过vgg-16，特征图变化如下：
conv1:300x300x3 -> 300x300x64
conv2:300x300x64 -> 150x150x128
conv3:150x150x128 -> 75x75x256
conv4:75x75x256 -> 38x38x512 --第一个尺度的特征图来源
conv5:38x38x512 -> 19x19x512

extra_layers前2层先进行分别3x3和1x1卷积，第2层输出一个尺度的特征图
conv6:19x19x512 -> 19x19x1024
conv7:19x19x1024 -> 19x19x1024 --第二个尺度的特征图
后8层分为4组，先通过1x1卷积，再通过步长为2的3x3卷积减半特征图的高和宽，每组的第二个卷积层的输出作为一个尺度的特征图来源
conv8:19x19x1024 -> 19x19x256 -> 10x10x512
conv9:10x10x512 -> 10x10x128 -> 5x5x256
conv9:5x5x256 -> 5x5x128 -> 3x3x256
conv9:3x3x256 -> 3x3x128 -> 1x1x256

对于上述6张不同尺度的特征图，每个像素点按照bounding_box给出的数目生成对应数量的锚框
'''
SSD300 = {
    'h': 300,
    'w': 300,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'steps': [8, 16, 32, 64, 100, 300],
    'scales': [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'clip': True,
}  # SSD300的配置参数


def BuildBaseNet(cfg, batch_norm=False):
    """
    构建基础网络vgg-16，用于目标检测的特征提取
    :param cfg: 列表形式表示的网络结构
    :param batch_norm: 是否使用BN层
    :return: 列表形式装载的网络层
    """
    torchvision.models.vgg16()
    layers = []  # 保存网络层
    in_channels = 3  # RGB图像输入通道为3
    for v in cfg:
        # 根据cfg中的元素构建网络结构
        if v == 'M':
            # 最大池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            # conv4-1前的池化层需要有ceil_mode=True，以使得75x75的特征图池化后变为38x38（否则为37x37）
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            # 卷积层和激活函数
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]  # vgg16第五个池化层修改为3x3-s1
    return layers


def BuildExtraNet(cfg, input_channels):
    """
    vgg后端添加的用于目标检测的额外网络层
    :param cfg: 列表形式表示的网络结构
    :param input_channels: 输入图像的通道数
    :return: 列表形式装载的网络层
    """
    layers = []  # 保存网络层
    in_channels = input_channels
    # conv6
    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=3, padding=1)]
    # conv7
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=1)]
    in_channels = cfg[1]
    flag = True  # 控制轮流出现1x1和3x3卷积
    # conv8 & conv9
    for v in cfg[2:6]:
        if flag:
            # 1x1卷积
            layers += [nn.Conv2d(in_channels, v, kernel_size=1)]
        else:
            # 3x3卷积，步长为2以减半高和宽
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=2)]
        flag = not flag
        in_channels = v
    # conv10 & conv11
    for v in cfg[6:]:
        if flag:
            # 1x1卷积
            layers += [nn.Conv2d(in_channels, v, kernel_size=1)]
        else:
            # 3x3卷积，无padding，步长默认
            layers += [nn.Conv2d(in_channels, v, kernel_size=3)]
        flag = not flag
        in_channels = v
    return layers


def BuildClassifier(cfg, base_net, extra_net, num_classes):
    """
    构建在不同尺度特征图上使用的预测器，选取的特征图在base_net和extra_net的指定位置
    :param cfg: 列表形式表示每个特征图上一个像素位置生成的锚框数量
    :param base_net: 基础网络各层的通道数
    :param extra_net: 额外网络各层的通道数
    :param num_classes: 目标类别数量
    :return: 列表形式装载的类别预测层和边界框预测层
    """
    cls_layers = []
    bbox_layers = []
    # 第一个特征图来自vgg-16网络的conv4-3层
    cls_layers += [BuildClassPredictor(base_net[conv4_3], cfg[0], num_classes)]
    bbox_layers += [BuildBboxPredictor(base_net[conv4_3], cfg[0])]
    # 后续五张特征图来自额外层的偶数层
    for k, v in enumerate(extra_net[1::2], 1):
        cls_layers += [BuildClassPredictor(v, cfg[k], num_classes)]
        bbox_layers += [BuildBboxPredictor(v, cfg[k])]
    return cls_layers, bbox_layers


def BuildClassPredictor(num_inputs, num_anchors, num_classes):
    """
    类别预测层
    目标类别数量为q，外加1个背景类，故锚框有q+1个类别
    若某个尺度的特征图大小为h*w，每个像素上生成a个锚框，则需要对h*w*a个锚框进行q+1分类，输出a*(q+1)个参数
    使用全连接层的参数过多，故使用卷积层的通道来输出类别预测，降低复杂度
    :param num_inputs: 输入通道数
    :param num_anchors: 锚框数量
    :param num_classes: 类别数量
    :return: 用于分类任务的卷积层
    """
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


def BuildBboxPredictor(num_inputs, num_anchors):
    """
    边界框预测层，为每个锚框预测4个偏移量，故输出通道是锚框数量的4倍
    :param num_inputs: 输入通道数
    :param num_anchors: 锚框数量
    :return: 用于预测预设框偏移量的卷积层
    """
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


class SSD(nn.Module):
    """
    SSD: Single Shot MultiBox Detector
    目标检测模型
    ------------------------------------------------------------
    通过特征提取网络提取图像特征，并增加额外网络结构获取不同尺度的特征图，
    在特征图上的每个位置生成预先设计好大小和尺寸的预设框，
    通过卷积层预测这些预设框的偏移量和所包围物体的种类，完成目标检测的过程。
    ------------------------------------------------------------
    其中，特征提取网络使用训练好的图像分类网络，如VGG-16
    边框预测被视为回归问题，采用Smooth L1损失函数
    类别预测为多分类问题，采用softmax损失函数（即使用softmax计算预测值概率分布的交叉熵损失函数）
    总损失为边框损失和分类损失的加权和
    """

    def __init__(self, num_classes, batch_norm=False, pretrained=True):
        """
        :param num_classes: 目标类别数量
        :param batch_norm: 训练时是否启用BN层
        """
        print('SSD init...')
        super(SSD, self).__init__()

        # 基础网络，提取图像特征
        print('\tbuild base net: ', end='')
        self.backbone = nn.ModuleList(BuildBaseNet(vgg16, batch_norm))
        if pretrained:
            self.LoadBaseNetWeights(vgg16_weights_file)
        print('completed')

        # 额外网络，获取多尺度特征图
        print('\tbuild extra net: ', end='')
        self.extra_net = nn.ModuleList(BuildExtraNet(extra_layers, vgg16[-1]))  # vgg-16的最后一层输出是额外层的输入
        print('completed')

        # 分类器，预测目标种类和锚框偏移量
        print('\tbuild classifier: ', end='')
        c, b = BuildClassifier(bounding_box, vgg16, extra_layers, num_classes)
        self.cls_layers = nn.ModuleList(c)
        self.bbox_layers = nn.ModuleList(b)
        print('completed')

        # 其他参数的设置
        print('\tconfigure parameters: ', end='')
        self.cfg = SSD300  # 模型参数配置
        self.num_classes = num_classes
        self.h = 300
        self.w = 300
        self.L2Norm = L2Norm(512, 20)  # 细粒度特征图的数值较大，通过l2标准化改变数值范围
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax()
        self.extra_net.apply(self.weights_init)
        self.cls_layers.apply(self.weights_init)
        self.bbox_layers.apply(self.weights_init)
        print('completed')

        # 生成预设框
        print('\tgenerate prior bounding box: ', end='')
        self.prior_boxes = GenerateBox(self.cfg['feature_maps'], self.cfg['h'], self.cfg['steps'],
                                       self.cfg['scales'], self.cfg['aspect_ratios'])
        print('completed')

        print('SSD init completed')

    def forward(self, x):
        features, classes_preds, bbox_preds = [], [], []  # 保存特征图、预设框、类别预测、预设框偏移量预测

        # 从conv4-3获取第一张特征图，通过L2标准化改变数值范围
        for i in range(23):
            x = self.backbone[i](x)
        features.append(self.L2Norm(x))

        # 继续正向传播至基础网络计算完毕
        for i in range(23, len(self.backbone)):
            x = self.backbone[i](x)

        # 额外网络层中，隔一层保存一次特征图
        flag = False
        for layer in self.extra_net:
            x = self.relu(layer(x))
            if flag:
                features.append(x)
            flag = not flag

        # 在每个特征图上预测锚框偏移量和目标类别
        for i in range(len(features)):
            # 获取每张特征图上的预测值，预测器返回的矩阵为[batch_size, channels, h, w]，把channels挪到最后一维
            classes_preds.append(self.cls_layers[i](features[i]).permute(0, 2, 3, 1).contiguous())
            bbox_preds.append(self.bbox_layers[i](features[i]).permute(0, 2, 3, 1).contiguous())

        # size(0)代表第0维的数据长度，元素i的0维是batch_size
        # 将预测值融合成一个维度，从而把各个特征图的预测矩阵拼接，变成[batch_size, 预测值]
        classes_preds = torch.cat([i.view(i.size(0), -1) for i in classes_preds], 1)
        bbox_preds = torch.cat([i.view(i.size(0), -1) for i in bbox_preds], 1)

        # 将预测值变为[batch_size, 锚框数, 目标类别数量]，[batch_size, 锚框数, 4(锚框坐标)]
        classes_preds = classes_preds.view(classes_preds.size(0), -1, self.num_classes + 1)
        bbox_preds = bbox_preds.view(bbox_preds.size(0), -1, 4)

        return self.prior_boxes.to(device), classes_preds.to(device), bbox_preds.to(device)

    def Predict(self, image, IOU_threshold=0.5, conf_threshold=0.01):
        self.eval()
        prior_boxes, cls_probs, offset_preds = self.forward(image.to(device))
        cls_probs = f.softmax(cls_probs, dim=2)
        object_preds = []
        for batch in range(cls_probs.shape[0]):
            predictions = prior_boxes, cls_probs[batch], offset_preds[batch]
            object_preds.append(Detection(predictions, IOU_threshold=IOU_threshold, conf_threshold=conf_threshold))
        return object_preds

    def LoadBaseNetWeights(self, weights_file):
        """
        加载基础网络的权重值
        :param weights_file: 权重文件路径
        :return:
        """
        base_net_weights = torch.load(weights_file)
        self.backbone.load_state_dict(base_net_weights)

    @staticmethod
    def xavier(param):
        # 一种初始化方式
        init.xavier_uniform_(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            # 判断m是否是卷积层
            self.xavier(m.weight.data)
            m.bias.data.zero_()

    def LoadPartWeights(self, weights, ignore):
        weights_dict = {k: v for k, v in weights.items() if k.find(ignore) == -1}
        model_dict = self.state_dict()
        model_dict.update(weights_dict)
        self.load_state_dict(model_dict)




