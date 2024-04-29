from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np 
import cv2

def bbox_iou(box1, box2):

	b1x1, b1y1, b1x2, b1y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
	b2x1, b2y1, b2x2, b2y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]	

	inter_rect_x1 = torch.max(b1x1, b2x1)
	inter_rect_y1 = torch.max(b1y1, b2y1)
	inter_rect_x2 = torch.max(b1x2, b2x2)
	inter_rect_y2 = torch.max(b1y2, b2y2)

	inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min = 0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min = 0)

	b1_area = (b1x2 - b1x1 + 1) * (b1y2 - b1y1 + 1)
	b2_area = (b2x2 - b2x1 + 1) * (b2y2 - b2y1 + 1)

	iou = inter_area(inter_area)/(b1_area + b2_area - inter_area)

	return iou


def unique(tensor):
	tensor_np = tensor.cpu().numpy()
	unique_np = np.unique(tensor_np)
	unique_tensor = torch.from_numpy(unique_np)

	tensor_res = tensor.new(unique_tensor.shape)
	tensor_res.copy_(unique_tensor)
	return tensor_res



def build_targets( pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim):
	nB = target.size(0)
	nA = num_anchors
	nC = num_classes
	nG = grid_size
	mask = torch.zeros(nB, nA, nG, nG)
	conf_mask = torch.ones(nB, nA, nG, nG)
	tx = torch.zeros(nB, nA, nG, nG)
	ty = torch.zeros(nB, nA, nG, nG)
	tw = torch.zeros(nB, nA, nG, nG)
	th = torch.zeros(nB, nA, nG, nG)
	tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
	tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

	nGT = 0
	nCorrect = 0
	for b in range(nB):
		for t in range(target.shape[1]):
			if target[b, t].sum() == 0:
			    continue
			nGT += 1
			gx = target[b, t, 1] * nG
			gy = target[b, t, 2] * nG
			gw = target[b, t, 3] * nG
			gh = target[b, t, 4] * nG
			gi = int(gx)
			gj = int(gy)
			gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
			anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
			anch_ious = bbox_iou(gt_box, anchor_shapes)
			conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
			best_n = np.argmax(anch_ious)
			gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
			pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
			mask[b, best_n, gj, gi] = 1
			conf_mask[b, best_n, gj, gi] = 1
			tx[b, best_n, gj, gi] = gx - gi
			ty[b, best_n, gj, gi] = gy - gj
			tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
			th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
			target_label = int(target[b, t, 0])
			tcls[b, best_n, gj, gi, target_label] = 1
			tconf[b, best_n, gj, gi] = 1

			iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
			pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
			score = pred_conf[b, best_n, gj, gi]
			if iou > 0.5 and pred_label == target_label and score > 0.5:
				nCorrect += 1

	return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls




def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)






def non_max_suppression(prediction, num_classes, conf_thres = 0.5, nms_thres = 0.4):
	box_corner = prediction.new(prediction.shape)
	box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2]/2
	box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3]/2
	box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2]/2
	box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3]/2
	prediciton[:, :, :4] = box_corner[:, :, :4]

	output = [None for _ in range(len(prediction))]
	for image_1, image_pred in enumerate(prediction):

		conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
		image_pred = image_pred[conf_mask]

		if not image_pred.size(0):
			continue
		class_conf, class_pred = torch.max(image_pred[:, 5:5+num_classes], 1, keepdim = True)

		detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()),1)
		unique_labels = detections[:, -1].cpu().unique()
		if prediction.is_cuda:
			unique_labels = unique_labels.cuda()

		for c in unique_labels:

			detection_class = detections[detections[:, -1] == c]

			_, conf_sort_index = torch.sort(detection_class[:, 4], descending = True)

			detection_class = detection_class[conf_sort_index]
			max_detections = []

			while detection_class.size(0):
				max_detections.append(detection_class[0].unsqueeze(0))

				if len(detection_class) == 1:
					break
				ious = bbox_iou(max_detections[-1], detection_class[1:])
				detection_class = detection_class[1:][ious <nms_thres]


			max_detections = torch.cat(max_detections).data

			output[image_1] = (max_detections if output[image_1] is None else torch.cat((output[image_1], max_detections)))


	return output
























