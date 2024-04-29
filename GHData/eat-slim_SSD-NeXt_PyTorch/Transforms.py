import platform
import time
import math
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as transforms
import cv2
import random
import matplotlib.pyplot as plt
import skimage
from skimage import util
import numpy as np
from BoundingBox import *
from DataSet import *
from BoundingBox import GetPositiveSamples


class MatchTarget(nn.Module):
    """
    读取数据阶段将真实框转换为损失函数所需的参数
    """

    def __init__(self, prior_boxes, IOU_threshold=0.5, approximate=False, cfg=None):
        """
        :param prior_boxes: 预设框
        :param IOU_threshold: 用于匹配的IOU阈值
        :param approximate: 匹配时是否为近似匹配
        :param cfg: 模型配置参数
        """
        super().__init__()
        self.prior_boxes = prior_boxes
        self.IOU_threshold = IOU_threshold
        self.approximate = approximate
        self.cfg = cfg

    def forward(self, targets):
        targets = GetPositiveSamples(self.prior_boxes, targets, self.IOU_threshold, self.approximate, self.cfg)
        return targets


class LetterBoxResize(nn.Module):
    """
    等比例缩放图片，空余部分用0填充，同时调整相应的真实框位置
    """

    def __init__(self, size=(288, 512), padding_ratio=0.5, normal_resize=0.0):
        """
        :param size: 缩放后的图片大小
        :param padding_ratio: 左（或上）端填充黑边的比例，0~1，-1时代表随机选取
        :param normal_resize: 以给定概率不考虑宽高比的拉伸至指定size
        """
        super().__init__()
        self.size = size
        self.padding_ratio = padding_ratio
        self.normal_resize = normal_resize
        self.resize = transforms.Resize(size=size)

    def forward(self, image, targets, image_only=False):
        if random.random() < self.normal_resize:
            return self.resize(image), targets

        # 计算宽和高的缩放比例，选择最小的那个进行等比例缩放，并在另一维度填充至指定size
        h, w = image.shape[-2], image.shape[-1]
        ratio_h = self.size[0] / h
        ratio_w = self.size[1] / w

        if ratio_h < ratio_w:
            ratio = ratio_h
            dim = 2  # 待填充的维度
        else:
            ratio = ratio_w
            dim = 1
        resize = transforms.Resize(size=(round(h * ratio), round(w * ratio)))
        image = resize(image)  # 等比例缩放
        padding_size = self.size[dim - 1] - image.shape[dim]

        if self.padding_ratio == -1:
            # 随机生成数字当做比例
            self.padding_ratio = random.random()

        padding_1 = round(padding_size * self.padding_ratio)
        padding_2 = padding_size - padding_1  # 计算两侧填充的数量
        if dim == 1:
            padding = torch.nn.ZeroPad2d((0, 0, padding_1, padding_2))
        else:
            padding = torch.nn.ZeroPad2d((padding_1, padding_2, 0, 0))
        image = padding(image)  # 用0填充

        if image_only:
            return image

        # 调整真实框坐标
        anchors = targets['anchors']
        if anchors.shape[0] > 0:
            if dim == 1:
                # 高度方向填充
                anchors[:, [1, 3]] = (padding_1 + anchors[:, [1, 3]] * round(h * ratio)) / self.size[0]
            else:
                # 宽度方向填充
                anchors[:, [0, 2]] = (padding_1 + anchors[:, [0, 2]] * round(w * ratio)) / self.size[1]
            anchors[:, :4].clamp_(max=1, min=0)
            targets['anchors'] = anchors
        return image, targets

    def Recover(self, size, anchors):
        h, w = size
        ratio_h = self.size[0] / h
        ratio_w = self.size[1] / w

        if ratio_h < ratio_w:
            ratio = ratio_h
            dim = 2  # 待填充的维度
        else:
            ratio = ratio_w
            dim = 1
        shape = (round(h * ratio), round(w * ratio))
        padding_size = self.size[dim - 1] - shape[dim - 1]

        padding_1 = round(padding_size * self.padding_ratio)

        if anchors.shape[0] > 0:
            if dim == 1:
                # 高度方向填充
                anchors[:, [1, 3]] = (anchors[:, [1, 3]] * self.size[0] - padding_1) / shape[0]
            else:
                # 宽度方向填充
                anchors[:, [0, 2]] = (anchors[:, [0, 2]] * self.size[1] - padding_1) / shape[1]
            anchors[:, :4].clamp_(max=1, min=0)

        return anchors


class RandomFlip(nn.Module):
    """
    随机翻转
    """

    def __init__(self, lr_ratio=0.5, ud_ratio=0.5):
        """
        :param lr_ratio: 左右翻转概率
        :param ud_ratio: 上下翻转概率
        """
        super().__init__()
        self.lr_ratio = lr_ratio
        self.ud_ratio = ud_ratio
        self.lr_flip = transforms.RandomHorizontalFlip(1)
        self.ud_flip = transforms.RandomVerticalFlip(1)

    def forward(self, image, targets):
        anchors = targets['anchors']
        if random.random() < self.lr_ratio:
            image = self.lr_flip(image)
            anchors[:, [0, 2]] = 1 - anchors[:, [2, 0]]  # 左右翻转，交换xmin和xmax
        if random.random() < self.ud_ratio:
            image = self.ud_flip(image)
            anchors[:, [1, 3]] = 1 - anchors[:, [3, 1]]  # 左右翻转，交换xmin和xmax
        targets['anchors'] = anchors
        return image, targets


class RandomCrop(nn.Module):
    """
    随机裁剪
    """

    def __init__(self, p=0.5, ratio=None):
        """
        :param p: 启用概率
        :param ratio: 元组(h, w)，裁剪后的图像高宽比尽可能贴近h/w，为None时不考虑
        """
        super().__init__()
        self.p = p
        self.ratio = ratio
        if self.ratio is not None:
            self.ratio = self.ratio[0] / self.ratio[1]

    def forward(self, image, targets):
        # 随机裁剪
        if random.random() < self.p:
            h, w = image.shape[-2], image.shape[-1]
            anchors = targets['anchors']

            # 得到可以包含所有真实框的最大范围
            max_bbox = torch.cat((torch.min(anchors[:, :2], dim=0)[0], torch.max(anchors[:, 2:4], dim=0)[0]), dim=0)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = 1 - max_bbox[2]
            max_d_trans = 1 - max_bbox[3]

            if self.ratio is None:
                # 在可选范围内随机选择裁剪后的范围
                crop_xmin = int(math.floor(max(0, (max_bbox[0] - random.uniform(0, max_l_trans))) * w))
                crop_ymin = int(math.floor(max(0, (max_bbox[1] - random.uniform(0, max_u_trans))) * h))
                crop_xmax = int(math.ceil(min(1, (max_bbox[2] + random.uniform(0, max_r_trans))) * w))
                crop_ymax = int(math.ceil(min(1, (max_bbox[3] + random.uniform(0, max_d_trans))) * h))
            else:
                # 尽可能贴近给定宽高比
                region_h = round(h * (max_bbox[3] - max_bbox[1]).item())
                region_w = round(w * (max_bbox[2] - max_bbox[0]).item())
                region_ratio = region_h / region_w  # 目标区域的高和宽之比
                if region_ratio < self.ratio:
                    # 高度较小，需要在高度方向额外保留背景
                    region_h = min(round(region_w * self.ratio), h)
                else:
                    # 宽度较小，需要在宽度方向额外保留背景
                    region_w = min(round(region_h / self.ratio), w)
                crop_ratio = min(1 - region_h / h, 1 - region_w / w)  # 保证高宽比不变时目标区域的扩大比率范围
                crop_ratio = random.uniform(0, crop_ratio) + 1  # 随机选择一个扩大比率，并计算相应扩大后的高和宽
                region_h = math.floor(region_h * crop_ratio)
                region_w = math.floor(region_w * crop_ratio)
                # 在范围内随机选择裁剪区域左上角位置，并计算出右下角位置
                # max(0, xmax-region_w) < x < min(xmin, w-region_w)
                crop_xmin = int(random.uniform(max(0, max_bbox[2] * w - region_w), min(max_bbox[0] * w, w - region_w)))
                crop_ymin = int(random.uniform(max(0, max_bbox[3] * h - region_h), min(max_bbox[1] * h, h - region_h)))
                crop_xmax = crop_xmin + region_w
                crop_ymax = crop_ymin + region_h

            # 更新图像
            image = image[:, crop_ymin: crop_ymax, crop_xmin: crop_xmax]
            new_h, new_w = image.shape[-2], image.shape[-1]

            # 更新真实框坐标
            anchors[:, [0, 2]] = ((anchors[:, [0, 2]] * w - crop_xmin) / new_w).clamp_(min=0, max=1)
            anchors[:, [1, 3]] = ((anchors[:, [1, 3]] * h - crop_ymin) / new_h).clamp_(min=0, max=1)
            targets['anchors'] = anchors
        return image, targets


class Mosaic(nn.Module):
    """
    随机用四张图片拼接成一张图片，其中一张图片为当前读取的图片，剩下三张从数据集中随机抽取
    """

    def __init__(self, dataset, offset=(0.5, 0.5), size=(288, 512), p=0.5):
        """
        :param dataset: 用于抽取图片的数据集
        :param offset: 划分线所在位置的范围
        :param size: 输出图像大小
        :param p: 启用概率
        """
        super().__init__()
        self.iter = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn, shuffle=True)
        self.offset = offset
        self.size = size
        self.p = p
        self.cropper = RandomCrop(p=1, ratio=size)  # 对参与拼接的图片进行裁剪，且保留所有目标区域

    def forward(self, image, target):
        if random.random() < self.p:
            # 在给定范围内随机选择划线范围
            offset_h = random.uniform(self.offset[0], self.offset[1])
            offset_w = offset_h
            pixel_h = round(self.size[0] * offset_h)  # 转换为像素值
            pixel_w = round(self.size[1] * offset_w)
            # 四个区域的高和宽像素值
            scales = [(pixel_h, pixel_w), (pixel_h, self.size[1] - pixel_w),
                      (self.size[0] - pixel_h, pixel_w), (self.size[0] - pixel_h, self.size[1] - pixel_w)]

            # 随机抽取三张图，与当前图片组合在一起
            images = [image]
            targets = [target['anchors']]
            img_id = target['image_id']
            cnt = 3
            for X, Y in self.iter:
                images.append(X[0])
                targets.append(Y['anchors'][0])
                cnt -= 1
                if cnt == 0:
                    break

            # 随机裁剪，保留较多目标区域，再将四张图片缩小至指定区域的大小
            for i in range(4):
                image, target, scale = images[i], targets[i], scales[i]
                image, target = self.cropper(image, {'anchors': target})
                target = target['anchors']

                h, w = image.shape[-2], image.shape[-1]
                ratio = min(scale[0] / h, scale[1] / w)
                resize = transforms.Resize(size=(round(h * ratio), round(w * ratio)))
                image = resize(image)
                images[i], targets[i] = image, target

            # 将四张图片放置在一张图上，且紧贴中心
            new_image = torch.zeros((3, self.size[0], self.size[1]), device=image.device)
            left_top = [(pixel_h - images[0].shape[-2], pixel_w - images[0].shape[-1]),
                        (pixel_h - images[1].shape[-2], pixel_w),
                        (pixel_h, pixel_w - images[2].shape[-1]),
                        (pixel_h, pixel_w)]  # 四张图片在新图的起始位置
            for i in range(4):
                image, target = images[i], targets[i]
                h, w = image.shape[-2], image.shape[-1]  # 图片自身高度和宽度
                h_min, w_min = left_top[i]  # 在新图的起始位置（原图左上角）
                h_max, w_max = h_min + h, w_min + w  # 在新图的末尾位置（原图右下角）
                new_image[:, h_min:h_max, w_min:w_max] = image
                target[:, [1, 3]] = (((target[:, [1, 3]] * h) + h_min) / self.size[0]).clamp_(min=h_min / self.size[0],
                                                                                              max=h_max / self.size[0])
                target[:, [0, 2]] = (((target[:, [0, 2]] * w) + w_min) / self.size[1]).clamp_(min=w_min / self.size[1],
                                                                                              max=w_max / self.size[1])
                targets[i] = target

            image = new_image
            target = {'anchors': torch.cat(targets, dim=0), 'image_id': img_id}
        return image, target


class RandomNoise(nn.Module):
    """
    在图像中随机增加噪声
    """

    def __init__(self, mode=['gaussian'], p=0.5):
        super().__init__()
        self.mode = mode
        self.p = p

    def forward(self, image):
        if random.random() < self.p:
            mode = random.choice(self.mode)
            image = np.array(image)
            image = util.random_noise(image, mode=mode).astype(np.float32)
        return image


class SSDCropping(nn.Module):
    """
    SSD原文对图像进行裁剪的方式
    每次从以下三种方式中随机选择
    1. 保留原图
    2. Random crop minimum IoU is among 0.1, 0.3, 0.5, 0.7, 0.9
    3. 随机裁剪
    参考: https://github.com/chauhan-utk/ssd.DomainAdaptation
    """

    def __init__(self):
        super().__init__()
        self.sample_options = (
            # Do nothing
            None,
            # min IoU, max IoU
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # no IoU requirements
            (None, None),
        )

    def forward(self, image, target):
        # Ensure always return cropped image
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:  # 不做随机裁剪处理
                return image, target

            htot, wtot = image.shape[-2], image.shape[-1]

            min_iou, max_iou = mode
            min_iou = float('-inf') if min_iou is None else min_iou
            max_iou = float('+inf') if max_iou is None else max_iou

            # Implementation use 5 iteration to find possible candidate
            for _ in range(5):
                # 0.3*0.3 approx. 0.1
                w = random.uniform(0.3, 1.0)
                h = random.uniform(0.3, 1.0)

                if w / h < 0.5 or w / h > 2:  # 保证宽高比例在0.5-2之间
                    continue

                # left 0 ~ wtot - w, top 0 ~ htot - h
                left = random.uniform(0, 1.0 - w)
                top = random.uniform(0, 1.0 - h)

                right = left + w
                bottom = top + h

                # boxes的坐标是在0-1之间的
                bboxes = target['anchors'][:, :-1]
                ious = IOU(bboxes, torch.tensor([[left, top, right, bottom]]))

                # 裁剪所有的真实框
                # all(): 所有张量均满足条件时返回True否则返回False
                if not ((ious > min_iou) & (ious < max_iou)).all():
                    continue

                # 去掉中心不在裁剪区域的真实框
                xc = 0.5 * (bboxes[:, 0] + bboxes[:, 2])
                yc = 0.5 * (bboxes[:, 1] + bboxes[:, 3])

                # 查找所有的gt box的中心点有没有在采样patch中的
                masks = (xc > left) & (xc < right) & (yc > top) & (yc < bottom)

                # 如果所有的gt box的中心点都不在采样的patch中，则重新找
                if not masks.any():
                    continue

                # 修改采样patch中的所有gt box的坐标（防止出现越界的情况）
                bboxes[bboxes[:, 0] < left, 0] = left
                bboxes[bboxes[:, 1] < top, 1] = top
                bboxes[bboxes[:, 2] > right, 2] = right
                bboxes[bboxes[:, 3] > bottom, 3] = bottom

                # 虑除不在采样patch中的gt box
                bboxes = bboxes[masks]

                # 获取在采样patch中的gt box的标签
                labels = target['anchors'][:, -1]
                labels = labels[masks]

                # 裁剪patch
                left_idx = int(left * wtot)
                top_idx = int(top * htot)
                right_idx = int(right * wtot)
                bottom_idx = int(bottom * htot)
                image = image[:, top_idx: bottom_idx, left_idx: right_idx]

                # 调整裁剪后的bboxes坐标信息
                bboxes[:, 0] = (bboxes[:, 0] - left) / w
                bboxes[:, 1] = (bboxes[:, 1] - top) / h
                bboxes[:, 2] = (bboxes[:, 2] - left) / w
                bboxes[:, 3] = (bboxes[:, 3] - top) / h

                # 更新crop后的gt box坐标信息以及标签信息
                target['anchors'] = torch.cat((bboxes, labels.unsqueeze(1)), dim=1)

                return image, target


class TensorForVGG(nn.Module):

    def __init__(self, mean=(123, 117, 104), size=(300, 300)):
        super().__init__()
        self.mean = torch.as_tensor(mean)
        self.size = size

    def forward(self, image):
        image = np.array(image)
        if self.size is not None:
            image = cv2.resize(image, self.size)
        image = torch.as_tensor(image.astype(np.float32)) - self.mean
        return image.permute(2, 0, 1)


class ComposeDefinedTrans:
    """
    重写Compose的__call__函数，满足调用自定义transforms的需求
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomApplyDefinedTrans(torch.nn.Module):
    """
    重写RandomApply的__call__函数，满足调用自定义transforms的需求
    """

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, img, target):
        if self.p < torch.rand(1):
            return img, target
        for t in self.transforms:
            img, target = t(img, target)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class RandomChoiceDefinedTrans:
    """
    重写RandomChoice的__call__函数，满足调用自定义transforms的需求
    """

    def __init__(self, trans, p=None):
        self.transforms = trans
        self.p = p

    def __call__(self, img, target):
        t = random.choices(self.transforms, weights=self.p)[0]
        return t(img, target)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if platform.system() == 'Windows':
        vocRoot = r'D:\DateSet\VOC'
    else:
        vocRoot = r'/root/autodl-tmp/VOC'
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # model = SSD_9_16(num_classes=20)
    h, w = 300, 300

    # 图片变换
    transforms_image = transforms.Compose([
        # transforms.RandomApply([transforms.ColorJitter(0.125, 0.5, 0.5, 0.05)], p=1),
        # RandomNoise(mode=['gaussian', 'localvar', 's&p', 'poisson', 'speckle'], p=0.8),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
    ])
    # 图片与标签变换
    transforms_all = RandomChoiceDefinedTrans([
        ComposeDefinedTrans([
            # SSDCropping(),
            LetterBoxResize(size=(h, w), padding_ratio=0.5, normal_resize=0),
            # RandomFlip(lr_ratio=0.5, ud_ratio=0)
        ]),
        Mosaic(
            VOCDetection(root_file=vocRoot, mode='trainval', dataset='KITTI',
                         transforms_image=transforms_image, transforms_all=RandomFlip(lr_ratio=0.5, ud_ratio=0)),
            offset=(0.5, 0.5), size=(h, w), p=1)
    ], p=[1, 0])

    train_dataset = VOCDetection(root_file=vocRoot, mode='trainval', transforms_image=transforms_image,
                                 transforms_all=transforms_all, dataset='KITTI')
    voc_iter = DataLoader(train_dataset, batch_size=1, collate_fn=train_dataset.collate_fn, shuffle=True)

    for X, Y in voc_iter:
        print(X.shape, Y['image_ids'])
        image = X[0].permute(1, 2, 0)
        anchors = Y['anchors'][0]
        anchors[:, -1] -= 1
        anchors = torch.cat((anchors, torch.ones((anchors.shape[0], 1))), dim=1)
        classes = train_dataset.classes
        print(anchors, anchors.shape)
        plt.xticks(alpha=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.tick_params(axis='y', width=0)
        display(image, anchors, 0.5, classes=classes, show_score=False)

        # 展示原图
        plt.xticks(alpha=0)
        plt.yticks(alpha=0)
        plt.tick_params(axis='x', width=0)
        plt.tick_params(axis='y', width=0)
        train_dataset.ShowImage(Y['image_ids'][0], show_score=False
                                # all_transforms=LetterBoxResize(size=(h, w), padding_ratio=0.5)
                                )

        time.sleep(1)
