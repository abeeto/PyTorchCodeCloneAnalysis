import torch 
import torch.nn as nn 
import numpy as np 

from utils import cal_iou
from anchor import Anchor
from retina_net import RetinaNet
from utils import UnNormalizer

cuda = torch.cuda.is_available()


class FocalLoss(nn.Module):
    '''
    Focal Loss
    '''
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__() 
        '''
        引用 'Focal Loss for Dense Object Detection' 论文原文：
        'As we will show in §5, we find that γ = 2 works well in practice and 
        the RetinaNet is relatively robust to γ ∈ [0.5, 5].'

        'Finally we note that α, the weight assigned to the rare class, also 
        has a stable range, but it interacts with γ making it necessary to select 
        the two together (see Tables 1a and 1b). In general α should be decreased 
        slightly as γ is increased (for γ = 2, α = 0.25 works best).'
        '''
        self.alpha = alpha 
        self.gamma = gamma 

    def forward(self, classification, localization, anchors, annotations):
        batch_size = classification.size()[0]
        cls_losses = []
        loc_losses = []

        for i in range(batch_size):
            pred_cls = classification[i, :, :]
            pred_loc = localization[i, :, :]
            anno = annotations[i, :, :]
            # 去掉为了batch中样本保持格式一致时，添加的[-1, -1, -1, -1, -1]
            anno = anno[anno[:, 0] != -1]

            # 首先考虑anno里什么都没有的情况
            if anno.size()[0] == 0:
                if cuda:
                    cls_losses.append(torch.tensor(0).float().cuda())
                    loc_losses.append(torch.tensor(0).float().cuda())
                else:
                    cls_losses.append(torch.tensor(0).float())
                    loc_losses.append(torch.tensor(0).float())
                continue
            '''
            如果要对数值进行 log 操作，最好先对其进行 clamp 操作，防止其中存在极小值，
            导致计算结果出现 nan。
            '''
            # 由于交叉熵要进行log()函数的运算，因此pred_cls中的数接近0或1时，
            # 会导致我们的交叉熵出现'nan'，因此需要将其限定在一定的范围内
            pred_cls = torch.clamp(pred_cls, min=1e-4, max=1-1e-4)

            iou = cal_iou(anchors[:, :], anno[:, :-1])

            # 计算各个anchor相对于annotations中，IoU最大的值，并记录最大IoU的位置
            # iou: [n, m]    iou_max: [n, ]
            iou_max, iou_max_ind = torch.max(iou, dim=1)

            '''
            计算 classification 的 loss 

            引用 'Focal Loss for Dense Object Detection' 论文原文：
            'Specifically, anchors are assigned to ground-truth object boxes 
            using an intersection-over-union (IoU) threshold of 0.5; and to 
            background if their IoU is in [0, 0.4). As each anchor is assigned 
            to at most one object box, we set the corresponding entry in its 
            length K label vector to 1 and all other entries to 0. If an anchor 
            is unassigned, which may happen with overlap in [0.4, 0.5), it is 
            ignored during training.'
            '''
            # 用于存储anchor被分配的类别，positive使用onehot编码(代码在后面实现)
            # negative使用全0编码，不参与训练的anchor使用全-1编码
            anchors_onehot = torch.ones(pred_cls.size()) * -1 # [-1, 80]
            if cuda:
                anchors_onehot = anchors_onehot.cuda()
            '''
            torch.lt(input, other, out=None)
            :逐元素比较input和other，即是否input < other
            :param input(Tensor): 要对比的张量
            :param other(Tensor or float): 对比的张量或float值
            :param out(Tensor,可选的): 输出张量
            '''
            # negative的anchor使用全0编码
            anchors_onehot[torch.lt(iou_max, 0.4), :] = 0
            '''
            torch.ge(input, other, out=None)
            :逐元素比较input和other，即是否input >= other。torch.gt()是判断input > other
            :param input(Tensor): 待对比的张量
            :prarm other(Tensor or float): 对比的张量或float值
            :param out(Tensor,可选的): 输出张量
            '''
            pos_ind = torch.ge(iou_max, 0.5)
            pos_num = pos_ind.sum()
            # print('positive anchor number:', pos_num)
            # 每一个anchor所属的GroundTruth位置和类别(按IoU计算结果分配), 
            # shape: [-1, 5],    [x1, y1, x2, y2, cls]
            gt = anno[iou_max_ind, :] 

            # positive的anchor使用onehot编码
            anchors_onehot[pos_ind, :] = 0
            anchors_onehot[pos_ind, gt[pos_ind, -1].long()] = 1

            if cuda:
                alpha = torch.ones(anchors_onehot.size()).cuda() * self.alpha
            else:
                alpha = torch.ones(anchors_onehot.size()) * self.alpha
            '''
            torch.where(condition, x, y) → Tensor
            针对于x而言，如果其中的每个元素都满足condition，就返回x的值；
            如果不满足condition，返回y的值。
            '''
            alpha = torch.where(anchors_onehot.eq(1), alpha, 1 - alpha)
            pt = torch.where(anchors_onehot.eq(1), pred_cls, 1 - pred_cls)
            # focal_weight = alpha(1-pt)^gamma
            focal_weight = alpha * torch.pow((1 - pt), self.gamma)
            # 交叉熵
            bce_loss = -1 * ( \
                anchors_onehot * torch.log(pred_cls) + \
                (1 - anchors_onehot) * torch.log(1 - pred_cls) \
                )
            cls_loss = focal_weight * bce_loss 
            # 不参与训练的anchor的loss需要从cls_loss中删除
            if cuda:
                cls_loss = torch.where(
                    torch.eq(anchors_onehot, -1), 
                    torch.zeros(cls_loss.size()).cuda(), 
                    cls_loss
                )
            else:
                cls_loss = torch.where(
                    torch.eq(anchors_onehot, -1), 
                    torch.zeros(cls_loss.size()), 
                    cls_loss
                )
            cls_losses.append(cls_loss.sum() / torch.clamp(pos_num.float(), min=1.0))
            # print(cls_losses) 

            '''
            计算 localization 的 loss 
            '''
            if pos_num <= 0:
                # 如果没有positive的anchor的话，我们将loc_loss置0
                if cuda:
                    loc_losses.append(torch.tensor(0).float().cuda())
                else:
                    loc_losses.append(torch.tensor(0).float().cuda())
            else:
                anchors_w = anchors[:, 2] - anchors[:, 0]
                anchors_h = anchors[:, 3] - anchors[:, 1]
                anchors_cx = anchors[:, 0] + 0.5 * anchors_w
                anchors_cy = anchors[:, 1] + 0.5 * anchors_h
                # 在BP算法中，我们只训练positive的anchor
                pos_w = anchors_w[pos_ind]
                pos_h = anchors_h[pos_ind]
                pos_cx = anchors_cx[pos_ind]
                pos_cy = anchors_cy[pos_ind]

                pos_gt = gt[pos_ind, :]
                gt_w = pos_gt[:, 2] - pos_gt[:, 0]
                gt_h = pos_gt[:, 3] - pos_gt[:, 1]
                gt_cx = pos_gt[:, 0] + 0.5 * gt_w
                gt_cy = pos_gt[:, 1] + 0.5 * gt_h 
                '''
                如果要对数值进行 log 操作，最好先对其进行 clamp 操作，防止其中存在极小值，
                导致计算结果出现 nan。
                '''
                # 同样的，我们在计算loc_loss时，依然会进行log()函数的运算，
                # 如果gt_w, gt_h过小的话，对导致最终输出的loc_loss为'nan'
                gt_w = torch.clamp(gt_w, min=1)
                gt_h = torch.clamp(gt_h, min=1)

                # 计算神经网络需要学习到的位置回归偏差
                dx = (gt_cx - pos_cx) / pos_w 
                dy = (gt_cy - pos_cy) / pos_h 
                dw = torch.log(gt_w / pos_w)
                dh = torch.log(gt_h / pos_h)

                d_stack = torch.stack((dx, dy, dw, dh)) 
                d_stack = d_stack.t() # 转置 
                '''
                引用 'Focal Loss for Dense Object Detection' 论文原文：
                'The training loss is the sum the focal loss and the standard 
                smooth L1 loss used for box regression [10].'
                因此，我们在计算loc_loss时，使用smooth L1 loss
                '''
                if cuda:
                    d_stack = d_stack / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    d_stack = d_stack / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]) 
                loc_loss = torch.abs(d_stack - pred_loc[pos_ind, :])
                '''
                torch.le(input, other, out=None)
                :逐元素比较input和other，即是否input <= other.
                :param input(Tenosr): 要对比的张量
                :param other(Tensor or float): 对比的张量或float值
                :param out(Tensor,可选的): 输出张量
                '''
                loc_loss = torch.where(
                    torch.le(loc_loss, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(loc_loss, 2),
                    loc_loss - 0.5 / 9.0
                )
                loc_losses.append(loc_loss.mean())
        
        cls_loss = torch.stack(cls_losses).mean(dim=0, keepdim=True)
        loc_loss = torch.stack(loc_losses).mean(dim=0, keepdim=True)
        # print(cls_loss)
        # print(loc_loss)

        return cls_loss, loc_loss

            
def imshow_postive_anchors(images, anchors, annotations):
    import matplotlib.pyplot as plt  
    import cv2

    batch_size = images.size()[0]
    for i in range(batch_size):
        image = images[i, :, :, :]
        anno = annotations[i, :, :]
        anno = anno[anno[:, 0] != -1]
        iou = cal_iou(anchors[:, :], anno[:, :-1])
        iou_max, iou_max_ind = torch.max(iou, dim=1)
        pos_ind = torch.ge(iou_max, 0.5)
        pos_anchors = anchors[pos_ind, :]
        print('positive anchor number:', pos_anchors.size())
    
        unnormalize = UnNormalizer()
        image = 255 * unnormalize(image)
        image = torch.clamp(image, min=0, max=255).data.numpy()
        image = np.transpose(image, (1, 2, 0)).astype(np.uint8)

        for x1, y1, x2, y2 in pos_anchors:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 1)

        image = image.get()
        print(image.shape)
        plt.figure() 
        image = image[:,:,[2,1,0]]
        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    data = np.load('./data/data_try.npy', allow_pickle=True).item() 
    print('image:', data['img'].size())
    print('annot:', data['annot'].size())
    print('scale:', data['scale'])

    retinanet = RetinaNet()
    classification, localization = retinanet(data['img'])
    print('classification:', classification.size())
    print('localization:', localization.size())

    anchor = Anchor() 
    total_anchors = anchor(data['img'])
    print('Anchors:', total_anchors.shape)

    focal_loss = FocalLoss()
    cls_loss, loc_loss = focal_loss(classification, localization, total_anchors, data['annot'])
    print('classification loss:', cls_loss)
    print('localization loss:', loc_loss)

    '''
    # 查看FocalLoss计算中positive anchors的位置
    '''
    # imshow_postive_anchors(data['img'], total_anchors[:, :], data['annot'])

    '''
    # ./data/data_try.npy 路径下保存的数据样本, 可以运行下面代码查看
    '''
    # from dataloader import imshow_img_anno
    # imshow_img_anno(data['img'][0], data['annot'][0])
    # imshow_img_anno(data['img'][1], data['annot'][1])