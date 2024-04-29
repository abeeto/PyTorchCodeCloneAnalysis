import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ANCHORS = (12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))

        return x


class ConvBNMish(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        padding = (kernel_size - 1) // 2

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            Mish()
        )


class ResBlock(nn.Module):
    """
    残差单元
    """

    def __init__(self, in_channels, res_num):
        """
        建立可学习参数
        :param in_channels: 输入通道数
        :param res_num: 残差结构的个数
        """
        super().__init__()

        self.res_num = res_num

        if res_num == 1:
            middle_channels = in_channels // 2
        else:
            middle_channels = in_channels

        self.conv = nn.Sequential()
        for index in range(res_num):
            self.conv.add_module(
                str(index),
                nn.Sequential(
                    ConvBNMish(in_channels, middle_channels, 1),
                    ConvBNMish(middle_channels, in_channels, 3)
                )
            )

    def forward(self, x):
        for index in range(self.res_num):
            x = x + self.conv[index](x)
        return x


class CSP(nn.Module):
    def __init__(self, in_channel, res_num):
        super().__init__()

        mid_channel = in_channel * 2

        if res_num != 1:
            res_channel = in_channel
            out_channel = res_channel * 2
        else:
            out_channel = res_channel = mid_channel

        # 首次进入 CSP 时, 特征图尺寸减半
        self.csp_in = ConvBNMish(in_channel, mid_channel, 3, 2)
        # 主线
        self.main = ConvBNMish(mid_channel, res_channel, 1, 1)
        # 旁路支线
        self.res = nn.Sequential(
            ConvBNMish(mid_channel, res_channel, 1, 1),
            ResBlock(res_channel, res_num),
            ConvBNMish(res_channel, res_channel, 1, 1)
        )

        self.csp_out = ConvBNMish(res_channel * 2, out_channel, 1, 1)

    def forward(self, x):
        x = self.csp_in(x)

        main = self.main(x)
        res = self.res(x)

        x = torch.cat((res, main), 1)

        x = self.csp_out(x)

        return x


class ConvBNLeakyReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        padding = (kernel_size - 1) // 2

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )


class SPP(nn.Module):
    """
    多尺度融合
    """

    def __init__(self, sizes=(5, 9, 13)):
        super().__init__()

        self.spp = nn.ModuleList(
            # 注意顺序, 先计算大尺寸池化
            [nn.MaxPool2d(size, 1, padding=size // 2) for size in sizes[::-1]]
        )

    def forward(self, x):
        pools = []
        for module in self.spp:
            pools.append(module(x))
        else:
            pools.append(x)

        x = torch.cat(pools, 1)

        return x


class UpSampleExpand(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()
        # -1 表示该维度保持不变
        x = x.view(B, C, H, 1, W, 1) \
            .expand(-1, -1, -1, self.stride, -1, self.stride) \
            .contiguous() \
            .view(B, C, H * self.stride, W * self.stride)

        return x


class SubNeckWithUpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.neck_in = ConvBNLeakyReLU(in_channels, in_channels // 2, 1)

        self.sample = UpSampleExpand(2)

        self.neck_branch = ConvBNLeakyReLU(in_channels, in_channels // 2, 1)

        self.neck_out = nn.Sequential(
            ConvBNLeakyReLU(in_channels, in_channels // 2, 1),
            ConvBNLeakyReLU(in_channels // 2, in_channels, 3),
            ConvBNLeakyReLU(in_channels, in_channels // 2, 1),
            ConvBNLeakyReLU(in_channels // 2, in_channels, 3),
            ConvBNLeakyReLU(in_channels, in_channels // 2, 1),
        )

    def forward(self, x, branch):
        x = self.neck_in(x)
        x = self.sample(x)
        branch = self.neck_branch(branch)

        x = torch.cat((branch, x), 1)
        x = self.neck_out(x)

        return x


class SubNeckNoUpSample(nn.Module):
    def __init__(self, main_channels, branch_channels):
        super().__init__()

        self.neck_in = ConvBNLeakyReLU(main_channels, branch_channels, 3, 2)

        self.neck_out = nn.Sequential(
            ConvBNLeakyReLU(branch_channels * 2, branch_channels, 1),
            ConvBNLeakyReLU(branch_channels, branch_channels * 2, 3),
            ConvBNLeakyReLU(branch_channels * 2, branch_channels, 1),
            ConvBNLeakyReLU(branch_channels, branch_channels * 2, 3),
            ConvBNLeakyReLU(branch_channels * 2, branch_channels, 1),
        )

    def forward(self, x, branch):
        x = self.neck_in(x)

        x = torch.cat((x, branch), 1)

        x = self.neck_out(x)

        return x


class YOLO(nn.Module):
    def __init__(self, out_shape, num_classes=80):
        super().__init__()

        self.anchors = [float(anchor) for anchor in ANCHORS]

        if out_shape[-1] == 76:
            self.anchor_mask = (0, 1, 2)
            self.scale_x_y = 1.2
        elif out_shape[-1] == 38:
            self.anchor_mask = (3, 4, 5)
            self.scale_x_y = 1.1
        elif out_shape[-1] == 19:
            self.anchor_mask = (6, 7, 8)
            self.scale_x_y = 1.05
        else:
            raise Exception("the shape of output not match")

        self.stride = 608 // out_shape[-1]
        self.shape = out_shape
        self.num_classes = num_classes

    def forward(self, x):
        """
        维度转换, 转换后的格式为 (B, 3*H*W, ...)
        :param x: 特征图输出 (B, 255, H, W)
        :return: 目标框与置信度
        """
        B, _, H, W = x.size()

        masked_anchors = []
        for mask in self.anchor_mask:
            # 乘以 2 表示每个 mask 占用 2 个 anchor, 总共有 9 个 mask 18 个 anchor
            masked_anchors += self.anchors[mask * 2:(mask + 1) * 2]

        masked_anchors = [anchor / self.stride for anchor in masked_anchors]

        anchor_groups = len(self.anchor_mask)

        anchor_w = []
        anchor_h = []

        x_list = []
        y_list = []
        w_list = []
        h_list = []
        det_confs_list = []
        cls_confs_list = []
        for group in range(anchor_groups):
            anchor_w.append(masked_anchors[group * 2])
            anchor_h.append(masked_anchors[group * 2 + 1])

            # 255 个通道可分为 3 组, 每一组的顺序为 X Y W H score num_class
            begin = group * (5 + self.num_classes)
            end = (group + 1) * (5 + self.num_classes)
            x_list.append(x[:, begin, :, :])
            y_list.append(x[:, begin + 1, :, :])
            w_list.append(x[:, begin + 2, :, :])
            h_list.append(x[:, begin + 3, :, :])
            det_confs_list.append(x[:, begin + 4, :, :])
            cls_confs_list.append(x[:, begin + 5:end, :, :])

        # 维度 (B, 3, H, W)
        x = torch.stack(x_list, 1)
        y = torch.stack(y_list, 1)
        w = torch.stack(w_list, 1)
        h = torch.stack(h_list, 1)

        det_confs = torch.stack(det_confs_list, 1).view(B, anchor_groups * H * W, 1)

        cls_confs = torch.cat(cls_confs_list, 1)
        cls_confs = cls_confs.view(B, anchor_groups, self.num_classes, H * W)
        cls_confs = cls_confs.permute((0, 1, 3, 2)).reshape(B, anchor_groups * H * W, self.num_classes)

        x = torch.sigmoid(x) * self.scale_x_y - 0.5 * (self.scale_x_y - 1)
        y = torch.sigmoid(y) * self.scale_x_y - 0.5 * (self.scale_x_y - 1)
        w = torch.exp(w)
        h = torch.exp(h)
        det_confs = torch.sigmoid(det_confs)
        cls_confs = torch.sigmoid(cls_confs)

        sequence = torch.arange(self.shape[-1], dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid([sequence, sequence])

        x = x + grid_x
        y = y + grid_y
        w = w * torch.tensor(anchor_w).view(1, -1, 1, 1)
        h = h * torch.tensor(anchor_h).view(1, -1, 1, 1)

        # 归一化
        x /= W
        y /= H
        w /= W
        h /= H

        # 组成 (x1, y1, x2, y2)
        x = x.view(B, anchor_groups * H * W, 1)
        y = y.view(B, anchor_groups * H * W, 1)
        w = w.view(B, anchor_groups * H * W, 1)
        h = h.view(B, anchor_groups * H * W, 1)

        x1 = x - 0.5 * w
        y1 = y - 0.5 * h
        x2 = x1 + w
        y2 = y1 + h

        boxes = torch.cat((x1, y1, x2, y2), 2)

        confs = det_confs * cls_confs

        return boxes, confs


class Darknet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # CSPDarknet53
        self.backbone = nn.ModuleList(
            [
                ConvBNMish(in_channels, 32, 3),
                CSP(32, 1),
                CSP(64, 2),
                CSP(128, 8),
                CSP(256, 8),
                CSP(512, 4)
            ]
        )

        # SPP+PAN
        self.spp = nn.Sequential(
            ConvBNLeakyReLU(1024, 512, 1),
            ConvBNLeakyReLU(512, 1024, 3),
            ConvBNLeakyReLU(1024, 512, 1),
            SPP(),
            ConvBNLeakyReLU(2048, 512, 1),
            ConvBNLeakyReLU(512, 1024, 3),
            ConvBNLeakyReLU(1024, 512, 1),
        )

        self.neck_76_76 = nn.ModuleList(
            [
                SubNeckWithUpSample(512),
                SubNeckWithUpSample(256),
                nn.Sequential(
                    ConvBNLeakyReLU(128, 256, 3),
                    nn.Conv2d(256, 255, 1, 1, bias=True),
                    YOLO(out_shape=(1, 255, 76, 76))
                )
            ]
        )

        self.neck_38_38 = nn.ModuleList(
            [
                SubNeckNoUpSample(128, 256),
                nn.Sequential(
                    ConvBNLeakyReLU(256, 512, 3),
                    nn.Conv2d(512, 255, 1, 1, bias=True),
                    YOLO(out_shape=(1, 255, 38, 38))
                )
            ]
        )

        self.neck_19_19 = nn.ModuleList(
            [
                SubNeckNoUpSample(256, 512),
                nn.Sequential(
                    ConvBNLeakyReLU(512, 1024, 3),
                    nn.Conv2d(1024, 255, 1, 1, bias=True),
                    YOLO(out_shape=(1, 255, 19, 19))
                )
            ]
        )

    def forward(self, x):
        backbone_76_76 = backbone_38_38 = None

        for module in self.backbone:
            x = module(x)
            if x.shape[-1] == 76:
                backbone_76_76 = x
            elif x.shape[-1] == 38:
                backbone_38_38 = x

        spp = self.spp(x)

        neck_1_1 = x_1 = self.neck_76_76[0](spp, backbone_38_38)
        neck_1_2 = x_1 = self.neck_76_76[1](x_1, backbone_76_76)
        boxes_1, confs_1 = self.neck_76_76[2](x_1)

        neck_2 = x_2 = self.neck_38_38[0](neck_1_2, neck_1_1)
        boxes_2, confs_2 = self.neck_38_38[1](x_2)

        x_3 = self.neck_19_19[0](neck_2, spp)
        boxes_3, confs_3 = self.neck_19_19[1](x_3)

        boxes = torch.cat((boxes_1, boxes_2, boxes_3), 1)
        confs = torch.cat((confs_1, confs_2, confs_3), 1)

        return boxes, confs

    def load_weights(self, weight_file):
        with open(weight_file, "rb") as fp:
            # 读取头部, 使文件指针偏移
            _ = np.fromfile(fp, dtype=np.int32, count=5)

            weights = np.fromfile(fp, dtype=np.float32)
            offset = 0

            for module in self.modules():
                # 没有偏置的卷积层, 其后为 BatchNorm 层
                if isinstance(module, (ConvBNMish, ConvBNLeakyReLU)):
                    conv = module[0]
                    bn = module[1]

                    # 加载 BatchNorm 层的参数
                    bn_buf_size = bn.bias.numel()
                    parameters = []
                    for _ in range(4):
                        parameter = torch.from_numpy(weights[offset:offset + bn_buf_size])
                        parameters.append(parameter)
                        offset += bn_buf_size
                    else:
                        bn_bias, bn_weight, bn_running_mean, bn_running_var = parameters

                    bn.bias.data.copy_(bn_bias)
                    bn.weight.data.copy_(bn_weight)
                    bn.running_mean.data.copy_(bn_running_mean)
                    bn.running_var.data.copy_(bn_running_var)
                # 带偏置的卷积层, 其后没有 BatchNorm 层
                elif isinstance(module, nn.Conv2d) and (not hasattr(module, "is_load")):
                    conv = module
                    conv_bias_size = conv.bias.numel()
                    conv.bias.data.copy_(torch.from_numpy(weights[offset:offset + conv_bias_size]))
                    offset += conv_bias_size
                else:
                    continue

                # 加载卷积层权重
                conv_weight_size = conv.weight.numel()
                conv_weight = torch.from_numpy(weights[offset:offset + conv_weight_size])
                offset += conv_weight_size

                conv.weight.data.copy_(conv_weight.reshape(conv.weight.data.shape))
                conv.is_load = True
