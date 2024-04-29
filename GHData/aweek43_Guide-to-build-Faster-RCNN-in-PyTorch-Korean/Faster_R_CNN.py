from RPN import RPN
from NMS import NMS
from Detector import Detector
from Loss import Loss

import numpy as np
import torch

sample_image = torch.zeros(1,3,224,224).float()
# sample ground-truth boxes 생성
ground_truth_boxes = np.asarray([[9, 6, 143, 114], [114, 86, 171, 143], [43,143,114,200]], dtype=np.float32) # format: [x1,y1,x2,y2]
# object label 임의로 지정
labels = np.asarray([6,8,2], dtype=np.int8)


ground_truth_boxes, labels, img_feature, anchors, pred_anchor_locs, pred_cls_scores, anchor_locations, anchor_labels, objectness_score  = RPN(sample_image, ground_truth_boxes, labels)
roi, pred_anchor_locs = NMS(anchors, pred_anchor_locs, objectness_score)
rpn_score, gt_rpn_score, rpn_loc, gt_rpn_loc, gt_roi_locs, gt_roi_labels, roi_cls_score, roi_cls_loc = Detector(ground_truth_boxes, labels, roi, img_feature, pred_anchor_locs, pred_cls_scores, anchor_locations, anchor_labels)
total_loss = Loss(rpn_score, gt_rpn_score, rpn_loc, gt_rpn_loc, gt_roi_locs, gt_roi_labels, roi_cls_score, roi_cls_loc)

print(total_loss)