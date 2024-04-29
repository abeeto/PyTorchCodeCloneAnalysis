import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import box_ops


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets):
        # bce loss
        bce_loss = F.binary_cross_entropy_with_logits(input=logits, 
                                                      target=targets, 
                                                      reduction="none"
                                                      )
        # We ignore those whose targets is -1.0. 
        pos_id = (targets==1.0).float()
        neg_id = (targets==0.0).float()
        pos_loss = pos_id * bce_loss
        neg_loss = neg_id * bce_loss
        loss = 1.0*pos_loss + 0.2*neg_loss

        if self.reduction == 'mean':
            batch_size = logits.size(0)
            loss = torch.sum(loss) / batch_size

        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


class GeneralIoULoss(nn.Module):
    def __init__(self, reduction):
        super(GeneralIoULoss, self).__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        """
        Both pred boxes and target boxes are expected in form (x1, y1, x2, y2), normalized by the image size.
        """
        B = pred_boxes.size(0)
        pred_boxes = pred_boxes.reshape(-1, 4)
        target_boxes = target_boxes.reshape(-1, 4)

        # compute giou
        giou = box_ops.giou_score(pred_boxes, target_boxes).reshape(B, -1)

        # loss
        loss = 1 - giou

        if self.reduction == 'mean':
            loss = loss.sum() / B

        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss


def compute_iou(anchor_boxes, gt_box):
    """
    Input:
        anchor_boxes : ndarray -> [[c_x_s, c_y_s, anchor_w, anchor_h], ..., [c_x_s, c_y_s, anchor_w, anchor_h]].
        gt_box : ndarray -> [c_x_s, c_y_s, anchor_w, anchor_h].
    Output:
        iou : ndarray -> [iou_1, iou_2, ..., iou_m], and m is equal to the number of anchor boxes.
    """
    # compute the iou between anchor box and gt box
    # First, change [c_x_s, c_y_s, anchor_w, anchor_h] ->  [x1, y1, x2, y2]
    # anchor box :
    ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # x1
    ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # y1
    ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # x2
    ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # y2
    w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]
    
    # gt_box : 
    # We need to expand gt_box(ndarray) to the shape of anchor_boxes(ndarray), in order to compute IoU easily. 
    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)

    gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2 # x1
    gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2 # y1
    gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2 # x2
    gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2 # y1
    w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

    # Then we compute IoU between anchor_box and gt_box
    S_gt = w_gt * h_gt
    S_ab = w_ab * h_ab
    I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
    I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])
    S_I = I_h * I_w
    U = S_gt + S_ab - S_I + 1e-20
    IoU = S_I / U
    
    return IoU


def set_anchors(anchor_size):
    """
    Input:
        anchor_size : list -> [[h_1, w_1], [h_2, w_2], ..., [h_n, w_n]].
    Output:
        anchor_boxes : ndarray -> [[0, 0, anchor_w, anchor_h],
                                   [0, 0, anchor_w, anchor_h],
                                   ...
                                   [0, 0, anchor_w, anchor_h]].
    """
    num_anchors = len(anchor_size)
    anchor_boxes = np.zeros([num_anchors, 4])
    for index, size in enumerate(anchor_size): 
        anchor_w, anchor_h = size
        anchor_boxes[index] = np.array([0, 0, anchor_w, anchor_h])
    
    return anchor_boxes


def multi_gt_creator(img_size, strides, label_lists, anchor_size, igt=0.5):
    """creator multi scales gt"""
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    h = w = img_size
    num_scale = len(strides)
    gt_tensor = []
    num_anchors = len(anchor_size) // num_scale

    for s in strides:
        gt_tensor.append(np.zeros([batch_size, h//s, w//s, num_anchors, 1+1+4+1]))
        
    # generate gt datas    
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            # get a bbox coords
            cls_id = int(gt_label[-1])
            x1, y1, x2, y2 = gt_label[:-1]
            # compute the center, width and height
            c_x = (x2 + x1) / 2 * w
            c_y = (y2 + y1) / 2 * h
            box_w = (x2 - x1) * w
            box_h = (y2 - y1) * h

            if box_w < 1. or box_h < 1.:
                # print('A dirty data !!!')
                continue    

            # compute the IoU
            anchor_boxes = set_anchors(anchor_size)
            gt_box = np.array([[0, 0, box_w, box_h]])
            iou = compute_iou(anchor_boxes, gt_box)

            # We only consider those anchor boxes whose IoU is more than ignore thresh,
            iou_mask = (iou > igt)

            if iou_mask.sum() == 0:
                # We assign the anchor box with highest IoU score.
                index = np.argmax(iou)
                # s_indx, ab_ind = index // num_scale, index % num_scale
                s_indx = index // num_anchors
                ab_ind = index - s_indx * num_anchors
                # get the corresponding stride
                s = strides[s_indx]
                # compute the gride cell location
                c_x_s = c_x / s
                c_y_s = c_y / s
                grid_x = int(c_x_s)
                grid_y = int(c_y_s)
                # compute gt labels
                weight = 2.0 - (box_w / w) * (box_h / h)

                if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1] = cls_id
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 2:6] = np.array([x1, y1, x2, y2])
                    gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = weight
            
            else:
                # There are more than one anchor boxes whose IoU are higher than ignore thresh.
                # But we only assign only one anchor box whose IoU is the best(objectness target is 1) and ignore other 
                # anchor boxes whose(we set their objectness as -1 which means we will ignore them during computing obj loss )
                # iou_ = iou * iou_mask
                
                # We get the index of the best IoU
                best_index = np.argmax(iou)
                for index, iou_m in enumerate(iou_mask):
                    if iou_m:
                        if index == best_index:
                            # s_indx, ab_ind = index // num_scale, index % num_scale
                            s_indx = index // num_anchors
                            ab_ind = index - s_indx * num_anchors
                            # get the corresponding stride
                            s = strides[s_indx]
                            # compute the gride cell location
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)

                            weight = 2.0 - (box_w / w) * (box_h / h)

                            if grid_y < gt_tensor[s_indx].shape[1] and grid_x < gt_tensor[s_indx].shape[2]:
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = 1.0
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 1] = cls_id
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 2:6] = np.array([x1, y1, x2, y2])
                                gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = weight
            
                        else:
                            # we ignore other anchor boxes even if their iou scores are higher than ignore thresh
                            # s_indx, ab_ind = index // num_scale, index % num_scale
                            s_indx = index // num_anchors
                            ab_ind = index - s_indx * num_anchors
                            s = strides[s_indx]
                            c_x_s = c_x / s
                            c_y_s = c_y / s
                            grid_x = int(c_x_s)
                            grid_y = int(c_y_s)
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 0] = -1.0
                            gt_tensor[s_indx][batch_index, grid_y, grid_x, ab_ind, 6] = -1.0

    gt_tensor = [gt.reshape(batch_size, -1, 1+1+4+1) for gt in gt_tensor]
    gt_tensor = np.concatenate(gt_tensor, 1)
    
    return gt_tensor


def loss(pred_obj, pred_cls, pred_box, targets):
    # loss func
    conf_loss_function = BCEWithLogitsLoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    reg_loss_function = GeneralIoULoss(reduction='none')

    # pred
    pred_obj = pred_obj[:, :, 0]
    pred_cls = pred_cls.permute(0, 2, 1)

    # gt    
    gt_obj = targets[:, :, 0].float()
    gt_cls = targets[:, :, 1].long()
    gt_box = targets[:, :, 2:6].float()
    gt_weight = targets[:, :, 6].float()
    gt_pos = (gt_weight > 0.).float()

    batch_size = pred_obj.size(0)
    # obj loss
    obj_loss = conf_loss_function(pred_obj, gt_obj)
    
    # cls loss
    cls_loss = (cls_loss_function(pred_cls, gt_cls) * gt_pos).sum() / batch_size
    
    # reg loss
    reg_loss = (reg_loss_function(pred_box, gt_box) * gt_weight * gt_pos).sum() / batch_size

    # total loss
    total_loss = obj_loss + cls_loss + reg_loss

    return obj_loss, cls_loss, reg_loss, total_loss


if __name__ == "__main__":
    gt_box = np.array([[0.0, 0.0, 10, 10]])
    anchor_boxes = np.array([[0.0, 0.0, 10, 10], 
                             [0.0, 0.0, 4, 4], 
                             [0.0, 0.0, 8, 8], 
                             [0.0, 0.0, 16, 16]
                             ])
    iou = compute_iou(anchor_boxes, gt_box)
    print(iou)
