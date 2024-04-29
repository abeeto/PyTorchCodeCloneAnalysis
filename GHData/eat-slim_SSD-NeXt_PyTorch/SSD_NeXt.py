import torch
from torch import nn
from torch.nn import functional as f
import torchvision.transforms as transforms
from BoundingBox import *
from ConvNeXt import convnext_tiny, LayerNorm, DropPath
import seaborn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

convnext_tiny_weights_file = r'weights/convnext_tiny_22k_1k_384.pth'

SSD_NeXt_noattention_cfg = {
    'h': 384,
    'w': 384,
    'multi_scale_layers': [384, 384],
    'bounding_box': [3, 3, 3, 3, 3],
    'feature_maps_c': [192, 384, 768, 384, 384],
    'feature_maps_h': [48, 24, 12, 6, 3],
    'feature_maps_w': [48, 24, 12, 6, 3],
    'attention': [False, False, False, False, False],
    'assign_priors': [[[0.0320, 0.0380], [0.0420, 0.0880], [0.0880, 0.0619]],
                      [[0.0680, 0.1520], [0.1520, 0.1060], [0.1120, 0.2160]],
                      [[0.2240, 0.1940], [0.1560, 0.3460], [0.3120, 0.3260]],
                      [[0.4880, 0.2200], [0.2440, 0.5080], [0.4040, 0.5520]],
                      [[0.7100, 0.3860], [0.6120, 0.7240], [0.9160, 0.6300]]],
    'style': True
}

SSD_NeXt_cfg = {
    'h': 384,
    'w': 384,
    'multi_scale_layers': [384, 384],
    'bounding_box': [3, 3, 3, 3, 3],
    'feature_maps_c': [192, 384, 768, 384, 384],
    'feature_maps_h': [48, 24, 12, 6, 3],
    'feature_maps_w': [48, 24, 12, 6, 3],
    'attention': [True, True, True, False, False],
    'assign_priors': [[[0.0320, 0.0380], [0.0420, 0.0880], [0.0880, 0.0619]],
                      [[0.0680, 0.1520], [0.1520, 0.1060], [0.1120, 0.2160]],
                      [[0.2240, 0.1940], [0.1560, 0.3460], [0.3120, 0.3260]],
                      [[0.4880, 0.2200], [0.2440, 0.5080], [0.4040, 0.5520]],
                      [[0.7100, 0.3860], [0.6120, 0.7240], [0.9160, 0.6300]]],
    'style': True
}


def ActivationFunc(style):
    if style:
        return nn.GELU()
    else:
        return nn.SiLU()


def NormLayer(style, channels):
    if style:
        return LayerNorm(channels, eps=1e-6, data_format="channels_first")
    else:
        return nn.BatchNorm2d(channels)


def BuildBaseNet(weights_file=None):
    """
    构建基础网络ConvNeXt-Tiny，使用其除最后两层以外的特征提取部分，用于目标检测的特征提取
    :param weights_file: 权重值文件位置
    :return: (ModuleList: 下采样层, ModuleList: ConvNeXt块)
    """
    convnext = convnext_tiny(1000)
    if weights_file is not None:
        convnext.load_state_dict(torch.load(weights_file)['model'])
    backbone = list(convnext.children())[:2]
    return nn.ModuleList([backbone[0], backbone[1]])


class FusionerBlock(nn.Module):
    """
    融合层块结构
    """

    def __init__(self, init_channels, kernel_size=3, padding=1, stride=1, layer_scale_init_value=1e-6, style=True):
        super().__init__()
        if style:
            self.block = nn.Sequential(
                nn.Conv2d(init_channels, 4 * init_channels, kernel_size=1),
                LayerNorm(4 * init_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(4 * init_channels, 4 * init_channels,
                          kernel_size=kernel_size, padding=padding, stride=stride, groups=4 * init_channels),
                nn.GELU(),
                nn.Conv2d(4 * init_channels, init_channels, kernel_size=1)
            )
            self.gamma = nn.Parameter(
                (layer_scale_init_value * torch.ones((init_channels,))).reshape(init_channels, 1, 1),
                requires_grad=True)
        else:
            self.block = nn.Sequential(
                nn.Conv2d(init_channels, 4 * init_channels, kernel_size=1),
                nn.BatchNorm2d(4 * init_channels),
                nn.SiLU(),
                nn.Conv2d(4 * init_channels, 4 * init_channels,
                          kernel_size=kernel_size, padding=padding, stride=stride, groups=4 * init_channels),
                nn.BatchNorm2d(4 * init_channels),
                nn.SiLU(),
                nn.Conv2d(4 * init_channels, init_channels, kernel_size=1),
                nn.BatchNorm2d(init_channels),
                nn.SiLU()
            )
            self.gamma = nn.Parameter(
                (torch.ones((init_channels,))).reshape(init_channels, 1, 1),
                requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.block(x)
        x = self.gamma * x
        x = shortcut + x
        return x


def BuildMultiScaleNeXt(cfg, input_channels, style=True):
    """
    ConvNeXt结构的多尺度网络
    :param cfg: 列表形式表示的网络结构
    :param input_channels: 输入图像的通道数
    :param style: 是否采用ConvNeXt网络风格
    :return: ModuleList装载的多尺度网络
    """
    if style:
        layers = nn.ModuleList([
            nn.Sequential(
                LayerNorm(input_channels, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(input_channels, cfg[0], kernel_size=2, stride=2),
                nn.GELU()
            ),
            nn.Sequential(
                LayerNorm(cfg[0], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(cfg[0], cfg[1], kernel_size=2, stride=2),
                nn.GELU()
            )
        ])
    else:
        layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, cfg[0], kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(cfg[0]),
                nn.SiLU()
            ),
            nn.Sequential(
                nn.Conv2d(cfg[0], cfg[1], kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(cfg[1]),
                nn.SiLU()
            )
        ])
    return layers


class SPP(nn.Module):
    """
    特征金字塔池化，融合多尺度目标
    """

    def __init__(self, input_channels, style=True):
        super().__init__()

        # 入口处将通道数变为输入的一半，出口处将通道数变为与输入相同的值
        self.enter = nn.Sequential(
            nn.Conv2d(input_channels, int(input_channels / 2), kernel_size=1),
            NormLayer(style, int(input_channels / 2)),
            ActivationFunc(style)
        )
        self.exit = nn.Sequential(
            nn.Conv2d(2 * input_channels, input_channels, kernel_size=1),
            NormLayer(style, input_channels),
            ActivationFunc(style)
        )
        self.pool = nn.MaxPool2d(kernel_size=5, padding=2, stride=1)

    def forward(self, x):
        part = self.enter(x)
        parts = [part]
        for _ in range(3):
            part = self.pool(part)
            parts.append(part)
        fusion = torch.cat(parts, dim=1)  # 通道叠加
        return self.exit(fusion)


class PANeXt(nn.Module):
    """
    采用ConvNeXt结构改进的PAN特征融合，同时包含FPN和PAN结构
    """

    def __init__(self, init_channels, init_height, init_width, depth=3, style=True):
        super().__init__()
        # 创建多个融合层，用于不同尺度特征图concat后充分融合
        self.depth = depth
        self.style = style
        self.fusioners = nn.ModuleList([self.BuildFusioner(init_channels, int(init_channels / 2)),
                                        self.BuildFusioner(int(init_channels / 2), int(init_channels / 4)),
                                        self.BuildFusioner(int(init_channels / 2), int(init_channels / 4)),
                                        self.BuildFusioner(init_channels, int(init_channels / 2)),
                                        self.BuildFusioner(init_channels * 2, init_channels)])

        # 上采样层
        self.up_samplers = nn.ModuleList([self.BuildUpSampler(init_channels, init_height, init_width),
                                          self.BuildUpSampler(int(init_channels / 2), init_height * 2, init_width * 2)])

        # 下采样层
        self.down_samplers = nn.ModuleList([self.BuildDownSampler(int(init_channels / 8), int(init_channels / 4)),
                                            self.BuildDownSampler(int(init_channels / 4), int(init_channels / 2)),
                                            self.BuildDownSampler(int(init_channels / 2), int(init_channels))])

    def forward(self, features):
        """
        双向特征融合，先由深到浅，再由浅到深
        :param features: 四张尺度由大到小的特征图组成的list
        :return: 三张尺度由大到小的、融合后的特征图组成的list
        """
        temp_features = [features[3]]  # 临时保存由深到浅融合后的特征层
        # 取深一层融合后的特征层，经过上采样后与骨干网络提取的原始特征层拼接、融合
        for i in range(2):
            temp_features.append(
                self.fusioners[i](
                    torch.cat((features[2 - i], self.up_samplers[i](temp_features[-1])), dim=1)
                )
            )

        fusion = [features[0]]  # 保存最终融合后的特征层
        # 取浅一层特征图，经过下采样后与深一层特征图拼接、融合
        for i in range(len(temp_features) - 1, -1, -1):
            fusion.append(
                self.fusioners[4 - i](
                    torch.cat((self.down_samplers[2 - i](fusion[-1]), temp_features[i]), dim=1)
                )
            )

        return fusion[1:]

    def BuildFusioner(self, input_channels, output_channels):
        """
        根据输入通道数构建融合层，先使用1x1卷积融合各个通道、还原通道数，再通过FusionerBlock充分融合
        :param input_channels: 输入通道数
        :param output_channels: 输出通道数
        :return: nn.Sequential装载的融合层
        """
        layers = [nn.Conv2d(input_channels, output_channels, kernel_size=1)]
        for i in range(self.depth):
            layers.append(FusionerBlock(init_channels=output_channels, kernel_size=3, padding=1, style=self.style))
        return nn.Sequential(*layers)

    def BuildUpSampler(self, input_channels, input_height, input_width):
        """
        上采样层，将深层特征的宽高扩大为原来的2倍，通道数缩小为原来的1/2
        :param input_channels: 输入通道数
        :param input_height: 特征图高度
        :param input_width: 特征图宽度
        :return: nn.Sequential装载的上采样层
        """
        return nn.Sequential(
            NormLayer(self.style, input_channels),
            transforms.Resize(size=(input_height * 2, input_width * 2)),
            nn.Conv2d(input_channels, int(input_channels / 2), kernel_size=1)
        )

    def BuildDownSampler(self, input_channels, output_channels):
        """
        下采样层，将浅层特征的宽高缩小为原来的1/2
        :param input_channels: 输入通道数
        :param output_channels: 输出通道数
        :return: nn.Sequential装载的下采样层
        """
        return nn.Sequential(
            NormLayer(self.style, input_channels),
            nn.Conv2d(input_channels, output_channels, kernel_size=2, stride=2)
        )


class Attention(nn.Module):
    """
    基于定位预测的注意力机制，辅助类别预测
    以定位预测结果矩阵为查询向量q，特征图矩阵为k和v，使用加性模型作为注意力评分函数
    """

    def __init__(self, query_channels, key_channels, value_channels, style=True):
        super(Attention, self).__init__()
        # 查询向量线性变换
        self.W_q = nn.Conv2d(query_channels, key_channels, kernel_size=1)

        # 注意力矩阵线性变换
        self.W_a = nn.Conv2d(value_channels, value_channels, kernel_size=1)
        self.ac = ActivationFunc(style)

    def forward(self, queries, keys, values):
        """
        :param queries: [batch_size, num_priors * 4, h, w]，查询向量，定位预测输出矩阵
        :param keys: [batch_size, key_channels, h, w]，键，特征图矩阵
        :param values: [batch_size, value_channels, h, w]，值，特征图矩阵
        :return: [batch_size, value_channels, h, w]，注意力矩阵
        """
        # 通道数转换为指定维度：[batch_size, hidden_channels, h, w]
        queries = self.W_q(queries)

        # 注意力评分：[batch_size, value_channels, h, w]
        scores = queries * keys

        # 通过sigmoid门控获取注意力分布
        weight = torch.sigmoid(scores)

        # 注意力分布映射到值上，获得注意力矩阵
        attention = weight * values

        return self.ac(self.W_a(attention))


class Predictor(nn.Module):
    """
    预测器，用于在特征图上进行定位预测和分类预测
    """

    def __init__(self, input_channels, num_priors, num_classes, attention=False, style=True):
        """
        :param input_channels: 特征图的通道数
        :param num_priors: 特征图上每个像素位置生成的预设框数量
        :param num_classes: 目标类别数量
        """
        super().__init__()
        self.at = attention
        # 3x3卷积扩大感受域
        self.enlarger = nn.Conv2d(input_channels, 2 * input_channels, kernel_size=3, padding=1)
        self.norm = NormLayer(style, 2 * input_channels)
        self.ac = ActivationFunc(style)

        # 边界框定位预测器和分类预测器
        self.bbox_predictor = nn.Conv2d(2 * input_channels, num_priors * 4, kernel_size=1)
        self.class_predictor = nn.Conv2d(2 * input_channels, num_priors * (num_classes + 1), kernel_size=1)

        # 注意力机制
        if attention:
            self.attention = Attention(key_channels=2 * input_channels, query_channels=num_priors * 4,
                                       value_channels=2 * input_channels, style=style)

    def forward(self, features):
        # 先使用3x3卷积扩大感受野
        features = self.enlarger(features)
        features = self.norm(features)
        features = self.ac(features)

        # 预测边界框定位信息
        bbox_preds = self.bbox_predictor(features)

        # 以边界框定位信息作为查询向量，特征图作为键和值，计算注意力分布，并映射到值，得到注意力分布加权后的特征图
        if self.at:
            features = self.attention(bbox_preds.detach(), features, features)

        # 在加权后的特征图上进行分类预测
        classes_preds = self.class_predictor(features)

        return bbox_preds, classes_preds


class SSD_NeXt(nn.Module):
    """
    SSD-NeXt
    基于SSD改进的目标检测模型，在整体架构不变的前提下，使用更新的技术
    ------------------------------------------------------------
    输入尺度：
    输入尺度由300×300改为更大的尺度384×384
    参考YOLOv3和v4的416×416尺度能够达到很高的检测速度，说明该量级尺度带来的开销仍是可控的
    ------------------------------------------------------------
    网络结构：
    骨干网络使用ConvNeXt-Tiny，该网络由facebook research提出，论文名称与来源如下
    `A ConvNet for the 2020s` —— https://arxiv.org/pdf/2201.03545.pdf
    ConvNeXt在ResNet的基础之上融合了多种网络结构的优点，并且使用与ViT相同的训练策略
    ConvNeXt-Tiny是一系列ConvNeXt网络中计算量、参数量最小的一个，能够保证很高的推理速度
    ConvNeXt-Tiny的特征提取网络结构，以及在本模型中得到的特征图尺度如下：
    input: [B, 3, 384, 384]
    ConvNeXt Block: covn_c,7x7-p3  LN  conv_4c,1x1  GELU  conv_c,1x1  Layer_Scale  Drop_Path
    Downsample: conv_2c,2x2-s2
    (0)conv96,4x4-s4 LN  TODO --out: [B, 96, 96, 96]
    (1)part 1: ConvNeXt Block(c=96) x3  TODO --out: [B, 96, 96, 96]
    (2)part 2: Downsample ConvNeXt Block(c=192) x3  TODO --out: [B, 192, 48, 48]
    (3)part 3: Downsample ConvNeXt Block(c=384) x9  TODO --out: [B, 384, 24, 24]
    (4)part 4: Downsample ConvNeXt Block(c=768) x3  TODO --out: [B, 768, 12, 12]

    使用SPP和PANeXt(包括了FPNeXt)结构进行特征融合，SPP与骨干网络的末尾相连，
    骨干网络最终的输出经过SPP获得P5特征图，与前三个ConvNeXt块的输出P2、P3、P4通过PANeXt进行双向特征融合
    PANeXt结构来自PANet，该网络将不同尺度特征图先由深到浅、再由浅到深双向融合，得到的最终特征图为：
        TODO --out1: [B, 192, 48, 48]  --out2: [B, 384, 24, 24]  --out3: [B, 768, 12, 12]

    多尺度特征网络减少为2块NeXt块，输入来自PANeXt得到的最深层特征图out3，通过两个卷积块得到的两张输出特征图为：
        TODO --out4: [B, 384, 6, 6]  --out5: [B, 384, 3, 3]
    ------------------------------------------------------------
    预测器：
    k-means聚类获取预设框，每层特征图均分配3个尺寸
    使用基于位置预测的注意力机制ALP，聚焦目标所在位置
    """

    def __init__(self, num_classes, cfg=None, weights_file=convnext_tiny_weights_file):
        """
        :param num_classes: 目标类别数量
        :param cfg: 配置参数
        """
        self.model_name = 'SSD-NeXt'
        print(f'{self.model_name} init...')
        super(SSD_NeXt, self).__init__()
        self.cfg = cfg  # 配置参数
        if cfg is None:
            self.cfg = SSD_NeXt_cfg
        self.num_classes = num_classes

        # 基础网络，提取图像特征
        print('\tbuild base net: ', end='')
        self.backbone = BuildBaseNet(weights_file)
        print('completed')

        # SPP层，融合多尺度目标
        print('\tbuild SPP: ', end='')
        self.spp = SPP(input_channels=self.cfg['feature_maps_c'][2], style=self.cfg['style'])
        print('completed')

        # PAN层，组合不同尺度特征图
        print('\tbuild PANeXt: ', end='')
        self.pan = PANeXt(init_channels=self.cfg['feature_maps_c'][2],
                          init_height=self.cfg['feature_maps_h'][2], init_width=self.cfg['feature_maps_w'][2],
                          style=self.cfg['style'])
        print('completed')

        # 额外网络，获取多尺度特征图
        print('\tbuild Multi Scale NeXt: ', end='')
        self.multi_scale_net = BuildMultiScaleNeXt(self.cfg['multi_scale_layers'], self.cfg['feature_maps_c'][2],
                                                   style=self.cfg['style']) \
            if len(self.cfg['multi_scale_layers']) > 0 else nn.ModuleList()
        print('completed')

        # 预测器，预测目标种类和锚框偏移量
        print('\tbuild predictors: ', end='')
        self.predictors = nn.ModuleList()
        for i in range(len(self.cfg['bounding_box'])):
            self.predictors.append(Predictor(self.cfg['feature_maps_c'][i], self.cfg['bounding_box'][i],
                                             num_classes, self.cfg['attention'][i], style=self.cfg['style']))
        print('completed')

        # 其他参数的设置
        print('\tconfigure parameters: ', end='')
        self.h = self.cfg['h']
        self.w = self.cfg['w']
        self.cfg['num_priors_per_map'] = \
            [self.cfg['bounding_box'][i] * self.cfg['feature_maps_h'][i] * self.cfg['feature_maps_w'][i]
             for i in range(len(self.cfg['bounding_box']))]
        print('completed')

        # 生成预设框
        print('\tgenerate prior bounding box: ', end='')
        self.prior_boxes = GenerateBoxAssigned(self.cfg['feature_maps_h'], self.cfg['feature_maps_w'],
                                               self.cfg['assign_priors'])
        print('completed')

        print(f'{self.model_name} {self.h}×{self.w} init completed')

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features, classes_preds, bbox_preds = [], [], []  # 保存特征图、预设框、类别预测、预设框偏移量预测

        # 提取特征，并从4个ConvNeXt块的输出获得4张不同粒度的特征图
        for i in range(4):
            x = self.backbone[0][i](x)  # 降采样
            x = self.backbone[1][i](x)  # 残差块
            features.append(x)

        # ConvNeXt最终输出经过SPP
        features[-1] = self.spp(features[-1])

        # PAN特征融合
        features = self.pan(features)

        # 从多尺度特征层获取不同尺度的特征图
        for layer in self.multi_scale_net:
            features.append(layer(features[-1]))

        # 在每个特征图上预测边界框偏移量和目标类别
        for i in range(len(features)):
            # 获取每张特征图上的预测值，预测器返回的矩阵为[batch_size, channels, h, w]，把channels挪到最后一维
            b, c = self.predictors[i](features[i])
            bbox_preds.append(b.permute(0, 2, 3, 1).contiguous())
            classes_preds.append(c.permute(0, 2, 3, 1).contiguous())

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

    def LoadPartWeights(self, weights, ignore):
        weights_dict = {k: v for k, v in weights.items() if k.find(ignore) == -1}
        model_dict = self.state_dict()
        model_dict.update(weights_dict)
        self.load_state_dict(model_dict)
