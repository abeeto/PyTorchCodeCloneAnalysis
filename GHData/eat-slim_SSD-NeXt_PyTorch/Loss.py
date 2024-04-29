import torch
import torch.nn as nn
import torch.nn.functional as f
from BoundingBox import *
from Argument import match_parameters



class MultiBoxLoss(nn.Module):
    """
    SSD损失函数
    MultiBoxLoss包括类别损失Lconf和位置损失Lloc两个部分，公式为
    L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
    其中，
    Lconf: softmax损失，即使用softmax函数获取目标概率分布的交叉熵损失，用于分类损失，
    Lloc: SmoothL1损失，用于定位损失
    alpha: 类别损失和位置损失加权和的权重值，根据交叉验证设置为1
    c: 类别标签
    l: 预测框
    g: 真实框
    N：参与损失计算的预设框的数量
    """

    def __init__(self, IOU_threshold=0.5, approximate=True, neg_ratio=3, alpha=1):
        """
        初始化损失函数
        :param IOU_threshold: 用于匹配的IOU阈值
        :param approximate: 是否近似匹配
        :param neg_ratio: 负例与正例的比值
        :param alpha:
        """
        super(MultiBoxLoss, self).__init__()
        self.IOU_threshold = IOU_threshold
        self.approximate = approximate
        self.neg_ratio = neg_ratio
        self.alpha = alpha
        self.location_loss = nn.SmoothL1Loss(reduction='none')
        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, targets):
        """
        计算损失
        :param predictions: 预测值，元组(预设框, 类别预测, 位置预测)
            预设框的矩阵形状：[预设框数量, 4]，4代表锚框的四个坐标，中心坐标形式
            分类预测的矩阵形状：[batch_size, 预设框数量, 类别数量]
            偏移量预测的矩阵形状：[batch_size, 预设框数量, 4]，4代表预测框的四个偏移量
        :param targets: 真实框和其对应的种类标签
            矩阵形状：[batch_size, 目标数量, 5]，5代表真实框的4个坐标和1个类别标签，对角坐标形式
        :return: loss值
        """
        prior_boxes, classes_preds, offset_preds = predictions  # 分别取预设框，分类预测，偏移量预测
        # batch_size, num_prior, num_classes = classes_preds.shape[0], classes_preds.shape[1], classes_preds.shape[2]

        # 获得偏移量的真实值、分类真实值、正样本掩码
        offset_targets, classes_targets, priors_mask = \
            GetPositiveSamples(prior_boxes.data, targets.data, self.IOU_threshold, self.approximate)

        # 通过类别判断的正样本掩码
        classes_mask = classes_targets > 0

        # 使用SmoothL1损失函数计算定位损失，正样本掩码使负例的预测值和真实值为0，不影响loss的最终值
        loc_loss = self.location_loss(offset_preds.transpose(1, 2), offset_targets.transpose(1, 2)).sum(dim=1)
        loc_loss = (classes_mask.float() * loc_loss).sum(dim=1)

        # 计算每个锚框的分类损失
        con = self.confidence_loss(classes_preds.transpose(1, 2), classes_targets)
        con_neg = con.clone()

        # 通过掩码保留参与计算的正负样本，其他样本不参与计算
        con_neg[classes_mask] = 0.0
        pos_num = classes_mask.sum(dim=1)

        # 按照confidence_loss降序排列 con_idx(Tensor: [N, 8732])
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # 用于损失计算的负样本数是正样本的3倍（在原论文Hard negative mining部分），
        # 但不能超过总样本数8732
        neg_num = torch.clamp(3 * pos_num, max=classes_mask.size(1)).unsqueeze(-1)
        neg_mask = torch.lt(con_rank, neg_num)  # (lt: <) Tensor [N, 8732]

        # confidence最终loss使用选取的正样本loss+选取的负样本loss
        conf_loss = (con * (classes_mask.float() + neg_mask.float())).sum(dim=1)

        # 总损失
        total_loss = self.alpha * loc_loss + conf_loss

        # 避免出现图像中没有真实框的情况，需要将分母设定一个最小值
        num_mask = torch.gt(pos_num, 0).float()  # 统计一个batch中的每张图像中是否存在正样本
        pos_num = pos_num.float().clamp(min=1e-6)  # 防止出现分母为零的情况
        ret = (total_loss * num_mask / pos_num).mean(dim=0)  # 只计算存在正样本的图像损失
        return ret


class MultiBoxLossNonMatch(nn.Module):
    """
    SSD损失函数，真实值参数不需要进行锚框匹配
    MultiBoxLoss包括类别损失Lconf和位置损失Lloc两个部分，公式为
    L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
    其中，
    Lconf: softmax损失，即使用softmax函数获取目标概率分布的交叉熵损失，用于分类损失，
    Lloc: SmoothL1损失，用于定位损失
    alpha: 类别损失和位置损失加权和的权重值，根据交叉验证设置为1
    c: 类别标签
    l: 预测框
    g: 真实框
    N：参与损失计算的预设框的数量
    """

    def __init__(self, neg_ratio=3, alpha=1, cfg=None):
        """
        初始化损失函数
        :param neg_ratio: 负例与正例的比值
        :param alpha: 平衡系数
        :param cfg: 模型配置参数
        """
        super(MultiBoxLossNonMatch, self).__init__()
        self.neg_ratio = neg_ratio
        self.alpha = alpha
        self.location_loss = nn.SmoothL1Loss(reduction='none')
        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')
        self.weighting = self.ScaleWeighting(cfg)

    def forward(self, predictions, targets):
        """
        计算损失
        :param predictions: 预测值，元组(预设框, 类别预测, 位置预测)
            预设框的矩阵形状：[预设框数量, 4]，4代表锚框的四个坐标，中心坐标形式
            分类预测的矩阵形状：[batch_size, 预设框数量, 类别数量]
            偏移量预测的矩阵形状：[batch_size, 预设框数量, 4]，4代表预测框的四个偏移量
        :param targets: 真实值，元组(偏移量真实值, 分类真实值, 正样本掩码)
            偏移量真实值的矩阵形状：[batch_size, 预设框数量, 4]，4代表四个偏移量
            分类真实值的矩阵形状：[batch_size, 预设框数量]
            正样本掩码的矩阵形状：[batch_size, 预设框数量, 4]
        :return: loss值
        """
        prior_boxes, classes_preds, offset_preds = predictions  # 分别取预设框，分类预测，偏移量预测
        offset_targets, classes_targets, priors_mask = targets  # 获得偏移量的真实值、分类真实值、正样本掩码

        # 通过类别判断的正样本掩码
        classes_mask = classes_targets > 0
        self.weighting = self.weighting[0].expand_as(classes_mask).to(classes_mask.device)

        # 使用SmoothL1损失函数计算定位损失，正样本掩码使负例的预测值和真实值为0，不影响loss的最终值
        loc_loss = self.location_loss(offset_preds.transpose(1, 2), offset_targets.transpose(1, 2)).sum(dim=1)
        loc_loss = (self.weighting * classes_mask.float() * loc_loss).sum(dim=1)

        # 计算每个锚框的分类损失
        con = self.confidence_loss(classes_preds.transpose(1, 2), classes_targets)
        con_neg = con.clone()

        # 通过掩码保留参与计算的正负样本，其他样本不参与计算
        con_neg[classes_mask] = 0.0
        pos_num = classes_mask.sum(dim=1)

        # 按照confidence_loss降序排列 con_idx(Tensor: [N, 8732])
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # 用于损失计算的负样本数是正样本的3倍（在原论文Hard negative mining部分），
        # 但不能超过总样本数8732
        neg_num = torch.clamp(3 * pos_num, max=classes_mask.size(1)).unsqueeze(-1)
        neg_mask = torch.lt(con_rank, neg_num)  # (lt: <) Tensor [N, 8732]

        # confidence最终loss使用选取的正样本loss+选取的负样本loss
        conf_loss = (con * (classes_mask.float() + neg_mask.float())).sum(dim=1)

        # 总损失
        total_loss = self.alpha * loc_loss + conf_loss

        # 避免出现图像中没有真实框的情况，需要将分母设定一个最小值
        num_mask = torch.gt(pos_num, 0).float()  # 统计一个batch中的每张图像中是否存在正样本
        pos_num = pos_num.float().clamp(min=1e-6)  # 防止出现分母为零的情况
        ret = (total_loss * num_mask / pos_num).mean(dim=0)  # 只计算存在正样本的图像损失
        return ret

    def ScaleWeighting(self, cfg):
        """
        反比例函数加权损失
        :param cfg: 模型参数字典
        """
        if cfg is None:
            weighting = torch.ones((1,), dtype=torch.float32)
        else:
            num_priors_per_map = cfg['num_priors_per_map']
            weighting = torch.tensor([])
            for i in range(len(num_priors_per_map)):
                weighting = torch.cat(
                    (weighting,
                     torch.full((num_priors_per_map[i],), match_parameters['weighting'] / (i + 1))),
                    dim=0
                )
        return weighting.unsqueeze(0)


