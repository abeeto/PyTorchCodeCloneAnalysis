import random
import torch
from math import sqrt as sqrt
from itertools import product as product
import matplotlib.pyplot as plt
import torchvision.ops
from Argument import encoding_parameters, match_parameters


def GenerateBox(feature_maps, size, steps, scales, aspect_ratios):
    """
    生成锚框
    tips：原文中设定预设框尺度的最小最大值为0.2和0.9（第6种），其余4种尺度均匀分布在0.2-0.9之间，还需要额外的第7种尺度
    这里尺度设置为，最小0.1，第2种尺度0.2，最大1.05（第7种），其余4种尺度均匀分布在0.2-1.05之间
    尺度设置为可调参数，可以根据数据集中目标尺度的分布特征进行合理修改，例如YOLO的尺度来源于数据集中目标尺度的k-means聚类
    :param feature_maps: 特征图尺寸
    :param size: 原始图像的尺寸
    :param steps: 特征图相对于原图的步长
    :param scales: 预设框的尺度
    :param aspect_ratios: 预设框的高宽比
    :return: torch张量类型的预设框集合
    """
    prior_boxes = []  # 保存生成的预设框
    for k, f in enumerate(feature_maps):
        # 枚举所有特征图的大小
        for i, j in product(range(f), repeat=2):  # itertools.product可以生成传入参数的所有排列组合，repeat=2代表迭代器的重复次数
            # 遍历特征图上所有像素点
            f_k = size / steps[k]
            # 确定锚框中心点的位置，在某个像素的中央
            cx = (j + 0.5) / f_k  # 将坐标归一化为相对于原图长宽的相对位置（0-1）
            cy = (i + 0.5) / f_k

            # 生成高宽比为1的锚框
            s_k = scales[k]
            prior_boxes += [cx, cy, s_k, s_k]

            # 生成更大一点的高宽比为1的锚框，s_k'=sqrt(s_k * s_(k+1))
            s_k_larger = sqrt(s_k * (scales[k + 1]))
            prior_boxes += [cx, cy, s_k_larger, s_k_larger]

            # 生成剩余几种尺寸的锚框
            for ar in aspect_ratios[k]:
                prior_boxes += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                prior_boxes += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
    # 转化为torch的张量类型
    output = torch.Tensor(prior_boxes).view(-1, 4)
    output.clamp_(max=1, min=0)  # 值限定为min与max之间
    return output


def GenerateBoxHW(feature_maps_h, feature_maps_w, scales, aspect_ratios):
    """
    生成以每个像素为中心具有不同形状的锚框
    :param feature_maps_h: 特征图的高
    :param feature_maps_w: 特征图的高
    :param scales: 预设框的尺度
    :param aspect_ratios: 预设框的高宽比
    :return: torch张量类型的预设框集合
    """
    prior_boxes = []  # 保存生成的预设框
    for k in range(len(feature_maps_h)):
        # 枚举所有特征图
        h, w = feature_maps_h[k], feature_maps_w[k]  # 当前特征图的高和宽
        ratio = w / h
        for i, j in product(range(h), range(w)):  # itertools.product可以生成传入参数的所有排列组合
            # 遍历特征图上所有像素点
            # 确定锚框中心点的位置，在某个像素的中央
            cx = (j + 0.5) / feature_maps_w[k]  # 将坐标归一化为相对于原图长宽的相对位置（0-1）
            cy = (i + 0.5) / feature_maps_h[k]

            # 生成高宽比为1的锚框
            s_k = scales[k]
            prior_boxes += [cx, cy, s_k * sqrt(1 / ratio), s_k / sqrt(1 / ratio)]

            # 生成更大一点的高宽比为1的锚框，s_k'=sqrt(s_k * s_(k+1))
            s_k_larger = sqrt(s_k * (scales[k + 1]))
            prior_boxes += [cx, cy, s_k_larger * sqrt(1 / ratio), s_k_larger / sqrt(1 / ratio)]

            # 生成剩余几种尺寸的锚框
            for ar in aspect_ratios[k]:
                prior_boxes += [cx, cy, s_k * sqrt(ar / ratio), s_k / sqrt(ar / ratio)]
                prior_boxes += [cx, cy, s_k * sqrt(1 / ar / ratio), s_k / sqrt(1 / ar / ratio)]
    # 转化为torch的张量类型
    output = torch.Tensor(prior_boxes).view(-1, 4)
    output.clamp_(max=1, min=0)  # 值限定为min与max之间
    return output


def GenerateBoxAssigned(feature_maps_h, feature_maps_w, assign_priors):
    """
    生成以每个像素为中心具有不同形状的锚框
    :param feature_maps_h: 特征图的高
    :param feature_maps_w: 特征图的高
    :param assign_priors: 列表设定的预设框的宽度和高度
    :return: torch张量类型的预设框集合
    """
    prior_boxes = []  # 保存生成的预设框
    for k in range(len(feature_maps_h)):
        # 枚举所有特征图
        h, w = feature_maps_h[k], feature_maps_w[k]  # 当前特征图的高和宽
        for i, j in product(range(h), range(w)):  # itertools.product可以生成传入参数的所有排列组合
            # 遍历特征图上所有像素点
            # 确定锚框中心点的位置，在某个像素的中央
            cx = (j + 0.5) / feature_maps_w[k]  # 将坐标归一化为相对于原图长宽的相对位置（0-1）
            cy = (i + 0.5) / feature_maps_h[k]

            for prior in assign_priors[k]:
                # 生成指定宽度和高度的预设框
                prior_boxes += [cx, cy, prior[0], prior[1]]

    # 转化为torch的张量类型
    output = torch.Tensor(prior_boxes).view(-1, 4)
    output.clamp_(max=1, min=0)  # 值限定为min与max之间
    return output


def BoxCornerToCenter(boxes):
    """
    从(xmin, ymin, xmax, ymax)转化为(cx, cy, w, h)
    :param boxes: 对角格式的锚框集合，[锚框数量, 4]
    :return: 中心格式的锚框集合，[锚框数量, 4]
    """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2,  # 锚框左上和右下的x和y分别相加，再除以2得到中心点坐标
                      boxes[:, 2:] - boxes[:, :2]), 1)  # 锚框右下与左上的x和y分别相减，得到宽度和高度


def BatchBoxCornerToCenter(boxes):
    """
    批量从(xmin, ymin, xmax, ymax)转化为(cx, cy, w, h)
    :param boxes: 对角格式的锚框集合，[batch_size, 锚框数量, 4]
    :return: 中心格式的锚框集合，[batch_size, 锚框数量, 4]
    """
    return torch.cat(((boxes[:, :, 2:] + boxes[:, :, :2]) / 2,  # 锚框左上和右下的x和y分别相加，再除以2得到中心点坐标
                      boxes[:, :, 2:] - boxes[:, :, :2]), dim=2)  # 锚框右下与左上的x和y分别相减，得到宽度和高度


def BoxCenterToCorner(boxes):
    """
    从(cx, cy, w, h)转化为(xmin, ymin, xmax, ymax)
    :param boxes: 中心格式的锚框集合，[锚框数量, 4]
    :return: 对角格式的锚框集合，[锚框数量, 4]
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,  # 中心坐标减去高和宽的一半，得到左上角坐标
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)  # 中心坐标加上高和宽的一半，得到右下角坐标


def BatchBoxCenterToCorner(boxes):
    """
    从(cx, cy, w, h)转化为(xmin, ymin, xmax, ymax)
    :param boxes: 中心格式的锚框集合，[锚框数量, 4]
    :return: 对角格式的锚框集合，[锚框数量, 4]
    """
    return torch.cat((boxes[:, :, :2] - boxes[:, :, 2:] / 2,  # 中心坐标减去高和宽的一半，得到左上角坐标
                      boxes[:, :, :2] + boxes[:, :, 2:] / 2), dim=2)  # 中心坐标加上高和宽的一半，得到右下角坐标


def IOU(boxes_a, boxes_b):
    """
    计算两个锚框集合之间，所有锚框组合的IOU，锚框需遵循对角坐标形式
    交并比（原文称Jaccard系数）IOU = A ∩ B / A ∪ B = A ∩ B / (S(A) + S(B) - A ∩ B)
    S(A)代表A锚框包围的面积
    在对角坐标形式下，A = [xa_min, ya_min, xa_max, ya_max]，B = [xb_min, yb_min, xb_max, yb_max]
    经过演算可知，A与B重叠部分的对角坐标为
    overlap_xy_min = max([xa_min, ya_min], [xb_min, yb_min])
    overlap_xy_max = min([xa_max, ya_max], [xb_max, yb_max])
    :param boxes_a: 锚框集合a，[锚框数量a, 4]
    :param boxes_b: 锚框集合b，[锚框数量b, 4]
    :return: a和b之间任意两锚框IOU组成的矩阵，[锚框数量a, 锚框数量b]，其中[i, j]表示a中第i个锚框与b中第j个锚框的IOU
    """
    # 获取集合大小
    size_a, size_b = boxes_a.size(0), boxes_b.size(0)

    # overlap_xy矩阵中第[i, j]个元素为，a中第i个锚框与b中第j个锚框交叉部分的左上或右下角坐标(size=2的一维矩阵)
    overlap_xy_min = torch.max(boxes_a[:, :2].unsqueeze(1).expand(size_a, size_b, 2),
                               boxes_b[:, :2].unsqueeze(0).expand(size_a, size_b, 2))
    overlap_xy_max = torch.min(boxes_a[:, 2:].unsqueeze(1).expand(size_a, size_b, 2),
                               boxes_b[:, 2:].unsqueeze(0).expand(size_a, size_b, 2))

    # 获取重叠部分的宽和高，且为非负数
    overlap_wh = torch.clamp((overlap_xy_max - overlap_xy_min), min=0)

    # 计算重叠部分、锚框a和锚框b的面积
    s_overlap = overlap_wh[:, :, 0] * overlap_wh[:, :, 1]
    s_a = ((boxes_a[:, 2] - boxes_a[:, 0]) *
           (boxes_a[:, 3] - boxes_a[:, 1])).unsqueeze(1).expand_as(s_overlap)  # [A,B]
    s_b = ((boxes_b[:, 2] - boxes_b[:, 0]) *
           (boxes_b[:, 3] - boxes_b[:, 1])).unsqueeze(0).expand_as(s_overlap)  # [A,B]

    return s_overlap / (s_a + s_b - s_overlap)  # 根据公式计算IOU


def BatchIOU(boxes_a, boxes_b):
    """
    计算两个锚框集合之间，所有锚框组合的IOU，锚框需遵循对角坐标形式
    交并比（原文称Jaccard系数）IOU = A ∩ B / A ∪ B = A ∩ B / (S(A) + S(B) - A ∩ B)
    S(A)代表A锚框包围的面积
    在对角坐标形式下，A = [xa_min, ya_min, xa_max, ya_max]，B = [xb_min, yb_min, xb_max, yb_max]
    经过演算可知，A与B重叠部分的对角坐标为
    overlap_xy_min = max([xa_min, ya_min], [xb_min, yb_min])
    overlap_xy_max = min([xa_max, ya_max], [xb_max, yb_max])
    :param boxes_a: 锚框集合a，[batch_size, 锚框数量a, 4]
    :param boxes_b: 锚框集合b，[batch_size, 锚框数量b, 4]
    :return: a和b之间任意两锚框IOU组成的矩阵，[batch_size, 锚框数量a, 锚框数量b]，
            其中[B, i, j]表示第B个batch里，a中第i个锚框与b中第j个锚框的IOU
    """
    # 获取集合大小
    batch_size, size_a, size_b = boxes_a.shape[0], boxes_a.shape[1], boxes_b.shape[1]

    # overlap_xy矩阵中第[B, i, j]个元素为，第B个batch中，a中第i个锚框与b中第j个锚框交叉部分的左上或右下角坐标(size=2的一维矩阵)
    overlap_xy_min = torch.max(boxes_a[:, :, :2].unsqueeze(2).expand(batch_size, size_a, size_b, 2),
                               boxes_b[:, :, :2].unsqueeze(1).expand(batch_size, size_a, size_b, 2))
    overlap_xy_max = torch.min(boxes_a[:, :, 2:].unsqueeze(2).expand(batch_size, size_a, size_b, 2),
                               boxes_b[:, :, 2:].unsqueeze(1).expand(batch_size, size_a, size_b, 2))

    # 获取重叠部分的宽和高，且为非负数
    overlap_wh = torch.clamp((overlap_xy_max - overlap_xy_min), min=0)

    # 计算重叠部分、锚框a和锚框b的面积
    s_overlap = overlap_wh[:, :, :, 0] * overlap_wh[:, :, :, 1]
    s_a = ((boxes_a[:, :, 2] - boxes_a[:, :, 0]) *
           (boxes_a[:, :, 3] - boxes_a[:, :, 1])).unsqueeze(2).expand_as(s_overlap)  # [A,B]
    s_b = ((boxes_b[:, :, 2] - boxes_b[:, :, 0]) *
           (boxes_b[:, :, 3] - boxes_b[:, :, 1])).unsqueeze(1).expand_as(s_overlap)  # [A,B]

    return s_overlap / (s_a + s_b - s_overlap)  # 根据公式计算IOU


def Match(prior_boxes, true_boxes, is_object, IOU_threshold=0.5, approximate=True, cfg=None):
    """
    将预设框与真实框匹配，获得正例预设框，匹配规则如下：
    1）每个真实框与和它IOU最大的预设框相匹配
    2）剩余未匹配的预设框与IOU > IOU_threshold的真实框相匹配
    这意味着每个真实框都可能有多个预设框与之匹配，成功匹配的预设框为正例，未匹配的为负例
    :param prior_boxes: 预设框集合，[batch_size, 预设框数量, 4]
    :param true_boxes: 真实框集合，[batch_size, 真实框数量, 4]
    :param is_object: 真实框掩码，[batch_size, 真实框数量]
    :param IOU_threshold: 用于匹配的IOU阈值
    :param approximate: 匹配时是否为近似匹配，近似匹配不能保证每个真实框与和它IOU最大的预设框相匹配，但匹配速度更快
    :param cfg: 模型配置参数
    :return: 匹配结果，[batch_size, 预设框数量]
    """
    batch_size, num_prior, device = prior_boxes.shape[0], prior_boxes.shape[1], prior_boxes.device

    # 预设框转为对角坐标形式，并扩充为batch_size个，留作后续匹配
    prior_boxes_corner = BatchBoxCenterToCorner(prior_boxes)
    iou = BatchIOU(prior_boxes_corner, true_boxes)  # 计算IOU，得到iou矩阵[batch_size, 预设框数量, 真实框数量]

    '''
    匹配规则1的优先度大于匹配规则2，理论计算时应先计算1再计算2，
    但在编码阶段应该先进行匹配规则2，然后由匹配规则1得到的结果直接覆盖在2上，从而获得更高的优先级
    '''
    # 获取每个预设框与各个真实框的最大iou，以及对应的真实框索引，[batch_size, 预设框数量]
    max_true_ious, max_true_ids = torch.max(iou, dim=2)

    # 找出预设框的最大iou小于阈值的预设框，作为未匹配到的预设框，将其匹配情况设为-1
    matched_priors = max_true_ious >= IOU_threshold  # 符合匹配要求的
    if cfg is not None:
        num_priors_per_map = cfg['num_priors_per_map']
        num_small_boxes = num_priors_per_map[0] + num_priors_per_map[1]
        # 小目标降低匹配的IOU阈值
        matched_small_priors = max_true_ious >= (IOU_threshold * match_parameters['small_boxes_iou_ratio'])
        matched_small_priors[num_small_boxes:] = False
        matched_priors += matched_small_priors

    nonmatched_priors = matched_priors == False
    max_true_ids[nonmatched_priors] = -1
    matched_boxes_map = max_true_ids  # 保存当前匹配情况

    '''
    根据匹配规则1进行匹配
    精确算法需通过迭代的过程逐次获取每个真实框与和它IOU最大的预设框，
    且已匹配的预设框不能参与下一轮迭代的计算，保证每个真实框都与不同预设框相匹配。

    近似算法直接通过max函数一次性求出矩阵中所有真实框匹配的预设框，可能造成多个真实框匹配到相同的预设框，
    但最终只能记录其中一个匹配情况，因而无法完全满足规则1。

    迭代的过程是精准算法，但耗时更长；
    近似算法通过矩阵的并行计算可以一次性得到所有结果，耗时更短，但匹配结果可能无法完全满足规则。
    '''
    if approximate:
        # 获取每个真实框与各个预设框的最大iou，以及对应的预设框索引，[batch_size, 真实框数量]
        max_prior_ious, max_prior_ids = torch.max(iou, dim=1)

        # 更新对应位置上的匹配情况
        # 对于非目标值，首先将匹配表向后扩充一位，再将非目标值匹配到的预设框索引设为-1（即刚刚扩充的那个），最后再将最后一位丢弃
        row = torch.arange(0, batch_size, device=device).unsqueeze(1)
        matched_boxes_map = torch.cat((matched_boxes_map, torch.zeros((batch_size, 1),
                                                                      dtype=matched_boxes_map.dtype,
                                                                      device=device)), dim=1)
        max_prior_ids[is_object == False] = -1
        true_ids = torch.arange(0, true_boxes.shape[1], device=device).repeat(batch_size, 1)
        matched_boxes_map[row, max_prior_ids] = true_ids
        matched_boxes_map = matched_boxes_map[:, :-1]

        # # 另一种简要的方法是，非目标值只会改变第一列，所以提前保存第一列，更新后再将第一列还原（会导致匹配到第一列的真实值也无法匹配）
        # temp = matched_boxes_map[:, 0].clone()
        # row = torch.arange(0, batch_size, device=device).unsqueeze(1)
        # true_ids = torch.arange(0, true_boxes.shape[1], device=device).repeat(batch_size, 1)
        # matched_boxes_map[row, max_prior_ids] = true_ids
        # matched_boxes_map[:, 0] = temp

    else:
        num_true = true_boxes.shape[1]  # 获取真实框数量
        sum_true = is_object.sum()  # 获取真实框总量
        iou *= is_object.unsqueeze(1)  # 将非目标的iou设为0
        num_per_batch = num_prior * num_true  # 每个batch的二维iou矩阵大小

        # 依次给每个真实框寻找最大iou的预设框，每匹配一个都需要将被匹配到的预设框剔除，保证所有真实框匹配的预设框不重复
        for _ in range(sum_true):
            max_prior_idxes = torch.argmax(iou)  # 找到矩阵中最大值，得到其索引，该索引为一维索引
            batch = int(max_prior_idxes / num_per_batch)  # 求出所在batch
            max_prior_idxes -= batch * num_per_batch  # 获得在该batch上的行列一维索引
            true_idx = (max_prior_idxes % num_true).long()  # 将索引转化为行列形式索引，找到对应的预设框与真实框的索引
            prior_idx = (max_prior_idxes / num_true).long()
            matched_boxes_map[batch][prior_idx] = true_idx  # 更新匹配情况
            iou[batch, :, true_idx] = -1  # 使用-1填充，代表剔除该行和列
            iou[batch, prior_idx, :] = -1

    return matched_boxes_map


def EncodeOffsets(prior_boxes, loc_targets):
    """
    获取预设值与真实值的差值，即偏移量，并对偏移量进行编码，使得偏移量的分布更易于拟合
    编码后的中心坐标为
    offset_xy = ((xy_b -xy _a) / wh_a - u_xy)) / sigma_xy
    宽和高为
    offset_wh = (log(wh_b / wh_a + eps) - u_wh)) / sigma_wh
    其中各项参数均为超参数，eps为防止log计算数值溢出的较小数
    :param prior_boxes: 预设框，[预设框数量, 4]
    :param loc_targets: 定位预测的目标值，[batch_size, 预设框数量, 4]
    :return: 编码后的偏移量，[batch_size, 预设框数量, 4]
    """
    u_xy, u_wh, sigma_xy, sigma_wh, eps, cardinal_eps, modified = encoding_parameters['u_xy'], \
                                                                  encoding_parameters['u_wh'], \
                                                                  encoding_parameters['sigma_xy'], \
                                                                  encoding_parameters['sigma_wh'], \
                                                                  encoding_parameters['eps'], \
                                                                  encoding_parameters['cardinal_eps'], \
                                                                  encoding_parameters['modified']
    loc_targets = BatchBoxCornerToCenter(loc_targets)  # 转化为中心坐标的形式

    # 根据公式编码偏移量
    offset_xy = ((loc_targets[:, :, :2] - prior_boxes[:, :, :2]) / prior_boxes[:, :, 2:] - u_xy) / sigma_xy
    if modified:
        # 当预设框的宽高比远小于真实值时，会使基数变为负数，需要限制基数不小于一个正数
        cardinal = (2 * torch.sqrt(prior_boxes[:, :, 2:] / (loc_targets[:, :, 2:] + eps)) - 1).clamp_(min=cardinal_eps)
        offset_wh = (0 - torch.log(cardinal) - u_wh) / sigma_wh
    else:
        offset_wh = (torch.log(eps + loc_targets[:, :, 2:] / prior_boxes[:, :, 2:]) - u_wh) / sigma_wh

    # 组合成中心坐标形式的偏移量
    offset = torch.cat([offset_xy, offset_wh], dim=2)
    return offset


def GetPositiveSamples(prior_boxes, targets, IOU_threshold, approximate=True, cfg=None):
    """
    根据预设框、真实框和标签，获取正样本
    :param prior_boxes: 预测框，[预设框数量, 4]，4代表锚框的四个坐标，中心坐标形式
    :param targets: 真实框和其对应的种类标签，[batch_size, 目标数量, 5]，5代表真实框的4个坐标和1个类别标签，对角坐标形式
    :param IOU_threshold: 用于匹配的IOU阈值
    :param approximate: 匹配时是否为近似匹配
    :param cfg: 模型配置参数
    :return: 元组(偏移量真实值，类别真实值，正样本掩码)，
        矩阵形状分别为[batch_size, 预设框数量, 4]，[batch_size, 预设框数量]，[batch_size, 预设框数量, 4]
    """
    batch_size, num_prior, device = targets.shape[0], prior_boxes.shape[0], prior_boxes.device
    prior_boxes = prior_boxes.repeat(batch_size, 1, 1)
    is_object = targets[:, :, -1] > 0  # 真实物体掩码

    # 将预设框与真实框进行匹配
    matched_boxes_map = Match(prior_boxes, targets[:, :, :-1], is_object, IOU_threshold, approximate, cfg)

    # 统计匹配到真实框的预设框[batch_size, 预设框数量]，用布尔值表示，并赋值扩充为[batch_size, 预设框数量, 4]，
    # 该参数为掩码，对应位为1时代表有效
    priors_mask = ((matched_boxes_map >= 0).float().unsqueeze(-1)).repeat(1, 1, 4)

    # 从匹配结果判定是否为正例（匹配到真实框的）预设框，获得判定掩码
    positive_priors_mask = matched_boxes_map >= 0

    # 获得这些正例所对应的目标值，非正例为默认值0，代表背景类
    # tips：二维以上tensor用tensor取值时，需要传入额外的tensor以取到相应维度
    row = torch.arange(0, batch_size, device=device).unsqueeze(1)
    classes_targets = targets[:, :, -1][row, matched_boxes_map] * positive_priors_mask
    loc_targets = targets[:, :, :-1][row, matched_boxes_map] * priors_mask

    # 计算并编码偏移量，负样本的偏移量被标记为0
    offset_targets = EncodeOffsets(prior_boxes, loc_targets) * priors_mask

    return offset_targets, classes_targets.long(), priors_mask


def DecodeOffsets(prior_boxes, offset_preds):
    """
    根据预测的偏移量获取预测锚框的坐标
    根据编码过程反向解码，得到解码规则如下
    xy_b = wh_a * (offset_xy * sigma_xy + u_xy) + xy_a
    wh_b = wh_a * e ^ (offset_wh * sigma_wh + u_wh)
    其中各项参数均为超参数，eps为防止log计算数值溢出的较小数
    :param prior_boxes: 预设框，[预设框数量, 4]
    :param offset_preds: 偏移量的预测值，[预设框数量, 4]
    :return: 解码后的预测框坐标，[预设框数量, 4]
    """
    u_xy, u_wh, sigma_xy, sigma_wh, eps, modified = encoding_parameters['u_xy'], \
                                                    encoding_parameters['u_wh'], \
                                                    encoding_parameters['sigma_xy'], \
                                                    encoding_parameters['sigma_wh'], \
                                                    encoding_parameters['eps'], \
                                                    encoding_parameters['modified']
    boxes_preds_xy = prior_boxes[:, 2:] * (offset_preds[:, :2] * sigma_xy + u_xy) + prior_boxes[:, :2]
    if modified:
        boxes_preds_wh = prior_boxes[:, 2:] * (4 / torch.pow(1 + torch.exp(-offset_preds[:, 2:] * sigma_wh - u_wh), 2))
    else:
        boxes_preds_wh = prior_boxes[:, 2:] * torch.exp(offset_preds[:, 2:] * sigma_wh + u_wh)
    boxes_preds = torch.cat((boxes_preds_xy, boxes_preds_wh), dim=1)
    return BoxCenterToCorner(boxes_preds)


def NMS(boxes_preds, class_ids, confidences, IOU_threshold=0.5):
    """
    非极大值抑制算法，根据置信度去除多余的重叠预测框
    :param boxes_preds: 预测框坐标，[预测框数量, 4]，对角坐标形式
    :param class_ids: 目标类别，[预测框数量]
    :param confidences: 置信度，[预测框数量]
    :param IOU_threshold: IOU阈值，高于此阈值被视为重叠
    :return: 保留的锚框的索引，[保留锚框数量]
    """
    class_boxes_preds = boxes_preds + class_ids.unsqueeze(1).expand_as(boxes_preds)
    keep = torchvision.ops.nms(class_boxes_preds, confidences, IOU_threshold)
    return keep


def Detection(predictions, IOU_threshold=0.45, conf_threshold=0.01, top_k=200):
    """
    根据预测结果转化为最终预测的目标锚框，由于每个batch最终输出的预测框数量不同，所以一次只单独处理一个batch
    :param predictions: 一个batch的预测值，元组(预设框, 类别预测, 位置预测)
            预设框的矩阵形状：[预设框数量, 4]，4代表锚框的四个坐标
            分类预测的矩阵形状：[预设框数量, 类别数量]
            偏移量预测的矩阵形状：[预设框数量, 4]，4代表预测框的四个偏移量
    :param IOU_threshold: IOU阈值，用于NMS算法去除多余锚框
    :param conf_threshold: 置信度阈值，置信度低于此阈值则视为背景类
    :param top_k: 输出目标数量的最大值，取置信度最高的top_k个目标作为输出
    :return: 最终预测到的目标锚框，[最终预测框数量, 6]，6代表预测框的4个坐标、1个类别标签、1个置信度
    """
    prior_boxes, classes_preds, offset_preds = predictions  # 分别取预设框，分类预测，偏移量预测

    # 根据预设框和偏移量，获得最终预测框的对角坐标形式
    bbox_preds = DecodeOffsets(prior_boxes, offset_preds).clamp_(min=0, max=1)

    # 每个预设框去除背景类预测后，选择置信度最高的类别作为最终预测的类别，取其置信度和类别索引
    classes_preds = classes_preds[:, 1:]
    confidences, class_idces = torch.max(classes_preds, 1)
    conf_mask = confidences > conf_threshold
    bbox_preds = bbox_preds[conf_mask]
    confidences = confidences[conf_mask]
    class_idces = class_idces[conf_mask]

    # 去除重叠框
    keep = NMS(bbox_preds, class_idces, confidences, IOU_threshold)
    if keep.shape[0] == 0:
        return torch.tensor([])
    bbox_preds = bbox_preds[keep]
    class_idces = class_idces[keep]
    confidences = confidences[keep]

    if bbox_preds.shape[0] > top_k:
        _, ids = torch.sort(confidences, dim=0, descending=True)
        ids = ids[:top_k]
        bbox_preds = bbox_preds[ids]
        class_idces = class_idces[ids]
        confidences = confidences[ids]

    return torch.cat((bbox_preds, class_idces.unsqueeze(1), confidences.unsqueeze(1)), dim=1)


def display(img, objects, threshold=0.5, classes=None, colors=None, line_width=1, show_score=True):
    """
    绘制目标框
    :param img: 待绘制原图像
    :param objects: 目标tensor，[N, 6] 6->(xmin, ymin, xmax, ymax, cls, conf)
    :param threshold: 置信度阈值
    :param classes: 类别列表，与类别标签对应
    :param colors: 可选颜色
    :param line_width: 目标框粗细
    :param show_score: 是否展示置信度
    """
    plt.rcParams['figure.dpi'] = 300
    if colors is None:
        colors = ['r', 'y', 'g', 'm', 'c', 'b']  # 颜色
    fig = plt.imshow(img)
    h, w = img.shape[0:2]

    # 滤除非目标
    mask = objects[:, 4] >= 0
    objects = objects[mask]

    # 滤除低置信度
    mask = objects[:, 5] >= threshold
    objects = objects[mask]

    # 参数转换
    bboxes = (objects[:, :4] * torch.tensor([w, h, w, h])).detach().numpy()  # 转化为像素坐标
    if classes is not None:
        labels = [classes[i] for i in objects[:, 4].int().tolist()]  # 转化为字符串形式的类别标签
        cls_colors = [colors[i % len(colors)] for i in objects[:, 4].int().tolist()]
    else:
        cls_colors = [random.choice(colors) for _ in range(objects.shape[0])]
    scores = objects[:, 5].tolist()

    # 绘制目标
    for obj in range(objects.shape[0]):
        bbox = bboxes[obj]
        color = cls_colors[obj]

        rect = plt.Rectangle(
            xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
            fill=False, edgecolor=color, linewidth=line_width
        )  # 边界框
        fig.axes.add_patch(rect)

        if classes is not None:
            label = labels[obj]
            text_color = 'w' if color == 'k' else 'k'
            if show_score:
                text = label + '={:.0f}%'.format(scores[obj] * 100)
            else:
                text = label
            fig.axes.text(rect.xy[0], rect.xy[1], text,
                          va='center', ha='center', fontsize=5, color=text_color, weight='bold',
                          bbox=dict(facecolor=color, lw=0, alpha=0.5))
    plt.show()

