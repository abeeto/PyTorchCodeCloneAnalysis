from yolov3 import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from data_read import *
from get_objectness_label import *
from prediction import *


# 挑选出跟真实框IoU大于阈值的预测框 pred_box [N,H,W,NUM_ANCHORS,4],gt_boxes是真实框位置信息
def get_iou_above_thresh_inds(pred_box, gt_boxes, iou_threshold):
    batchsize = pred_box.shape[0]
    num_rows = pred_box.shape[1]
    num_cols = pred_box.shape[2]
    num_anchors = pred_box.shape[3]
    ret_inds = np.zeros([batchsize, num_rows, num_cols, num_anchors])
    for i in range(batchsize):
        pred_box_i = pred_box[i]  # [11,11,3,4]
        gt_boxes_i = gt_boxes[i]  # [50,4]
        for k in range(len(gt_boxes_i)):  # gt in gt_boxes_i:
            gt = gt_boxes_i[k]
            gtx_min = gt[0] - gt[2] / 2.
            gty_min = gt[1] - gt[3] / 2.
            gtx_max = gt[0] + gt[2] / 2.
            gty_max = gt[1] + gt[3] / 2.
            if (gtx_max - gtx_min < 1e-3) or (gty_max - gty_min < 1e-3):
                continue
            x1 = np.maximum(pred_box_i[:, :, :, 0], gtx_min)
            y1 = np.maximum(pred_box_i[:, :, :, 1], gty_min)
            x2 = np.minimum(pred_box_i[:, :, :, 2], gtx_max)
            y2 = np.minimum(pred_box_i[:, :, :, 3], gty_max)
            intersection = np.maximum(x2 - x1, 0.) * np.maximum(y2 - y1, 0.)
            s1 = (gty_max - gty_min) * (gtx_max - gtx_min)
            s2 = (pred_box_i[:, :, :, 2] - pred_box_i[:, :, :, 0]) * (pred_box_i[:, :, :, 3] - pred_box_i[:, :, :, 1])
            union = s2 + s1 - intersection
            iou = intersection / union
            above_inds = np.where(iou > iou_threshold)  # above_inds为iou中大于iou_threshold的数值组合
            ret_inds[i][above_inds] = 1
    ret_inds = np.transpose(ret_inds, (0, 3, 1, 2))
    return ret_inds.astype('bool')


def label_objectness_ignore(label_objectness, iou_above_thresh_indices):
    # 注意：这里不能简单的使用 label_objectness[iou_above_thresh_indices] = -1，
    #         这样可能会造成label_objectness为1的那些点被设置为-1了
    #         只有将那些被标注为0，且与真实框IoU超过阈值的预测框才被标注为-1
    negative_indices = (label_objectness < 0.5)
    ignore_indices = negative_indices * iou_above_thresh_indices
    label_objectness[ignore_indices] = -1
    return label_objectness


# reader = data_loader('./insects/train', batch_size=2, mode='train')
# img, gt_boxes, gt_labels, im_shape = next(reader())
# label_objectness, label_location, label_classification, scale_location = get_objectness_label(
#     img, gt_boxes, gt_labels, iou_threshold=0.7, anchors=[116, 90, 156, 198, 373, 326], num_classes=7, downsample=32
# )
# NUM_ANCHORS = 3
# NUM_CLASS = 7
# num_filters = NUM_ANCHORS * (NUM_CLASS + 5)
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# with torch.no_grad():
#     backbone = DarkNet53_conv_body().to(device)
#     detection = YoloDetectionBlock(ch_in=1024, ch_out=512).to(device)
#     conv2d_pred = torch.nn.Conv2d(1024, num_filters, 1, padding=0).to(device)
#     img = torch.from_numpy(img).to(device)
#     c0, c1, c2 = backbone(img)
#     route, tip = detection(c0)
#     p0 = conv2d_pred(tip)
#
#     anchors = [116, 90, 156, 198, 373, 326]
#     pred_boxes = get_yolo_box_xxyy(p0.cpu().numpy(), anchors, num_classes=7, downsample=32)
#     iou_above_thresh_inds = get_iou_above_thresh_inds(pred_boxes, gt_boxes, iou_threshold=0.7)
#     label_objectness = label_objectness_ignore(label_objectness, iou_above_thresh_inds)
#     print(label_objectness.shape)

def get_loss(output, label_objectness, label_location, label_classification, scales, num_anchors=3, num_classes=7):
    # 将output从[N, C, H, W]变形为[N, NUM_ANCHORS, NUM_CLASSES + 5, H, W]
    reshaped_output = torch.reshape(output, [-1, num_anchors, num_classes + 5, output.shape[2], output.shape[3]])

    # 从output中取出跟objectness相关的预测值
    pred_objectness = reshaped_output[:, :, 4, :, :]
    loss_function1 = nn.BCELoss().to(device)
    loss_function2 = nn.MSELoss().to(device)

    loss_objectness = loss_function2(pred_objectness, label_objectness)
    ## 对第1，2，3维求和
    # loss_objectness = fluid.layers.reduce_sum(loss_objectness, dim=[1,2,3], keep_dim=False)

    # pos_samples 只有在正样本的地方取值为1.，其它地方取值全为0.
    pos_objectness = label_objectness > 0
    pos_samples = pos_objectness.float()
    pos_samples.stop_gradient = True

    # 从output中取出所有跟位置相关的预测值
    tx = reshaped_output[:, :, 0, :, :]
    ty = reshaped_output[:, :, 1, :, :]
    tw = reshaped_output[:, :, 2, :, :]
    th = reshaped_output[:, :, 3, :, :]

    # 从label_location中取出各个位置坐标的标签
    dx_label = label_location[:, :, 0, :, :]
    dy_label = label_location[:, :, 1, :, :]
    tw_label = label_location[:, :, 2, :, :]
    th_label = label_location[:, :, 3, :, :]
    # 构建损失函数
    loss_location_x = loss_function2(tx, dx_label)
    loss_location_y = loss_function2(ty, dy_label)
    loss_location_w = loss_function2(tw, tw_label)
    loss_location_h = loss_function2(th, th_label)

    # 计算总的位置损失函数
    loss_location = loss_location_x + loss_location_y + loss_location_h + loss_location_w

    # 乘以scales
    # loss_location = loss_location * scales
    # 只计算正样本的位置损失函数
    # loss_location = loss_location * pos_samples

    # 从ooutput取出所有跟物体类别相关的像素点
    pred_classification = reshaped_output[:, :, 5:5 + num_classes, :, :]
    # 计算分类相关的损失函数
    loss_classification = loss_function2(pred_classification, label_classification)
    # 将第2维求和
    # loss_classification = torch.sum(loss_classification, dim=2, keepdim=False)
    # 只计算objectness为正的样本的分类损失函数
    # loss_classification = loss_classification * pos_samples
    total_loss = loss_objectness + loss_location + loss_classification
    # 对所有预测框的loss进行求和
    # total_loss = torch.sum(total_loss, dim=[1, 2, 3], keepdim=False)
    # 对所有样本求平均
    # total_loss = torch.mean(total_loss)

    return total_loss


scales = [2., 4.]
NUM_ANCHORS = 3
NUM_CLASS = 7
num_filters = NUM_ANCHORS * (NUM_CLASS + 5)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

reader = data_loader('./insects/train', batch_size=2, mode='train')
img, gt_boxes, gt_labels, im_shape = next(reader())
label_objectness, label_location, label_classification, scale_location = get_objectness_label(img,
                                                                                              gt_boxes, gt_labels,
                                                                                              iou_threshold=0.7,
                                                                                              anchors=[116, 90, 156,
                                                                                                       198, 373, 326],
                                                                                              num_classes=7,
                                                                                              downsample=32)

with torch.no_grad():
    backbone = DarkNet53_conv_body().to(device)
    detection = YoloDetectionBlock(ch_in=1024, ch_out=512).to(device)
    conv2d_pred = torch.nn.Conv2d(1024, num_filters, 1, padding=0).to(device)
    img = torch.from_numpy(img).to(device)
    c0, c1, c2 = backbone(img)
    route, tip = detection(c0)
    p0 = conv2d_pred(tip)

    anchors = [116, 90, 156, 198, 373, 326]
    pred_boxes = get_yolo_box_xxyy(p0.cpu().numpy(), anchors, num_classes=7, downsample=32)
    iou_above_thresh_inds = get_iou_above_thresh_inds(pred_boxes, gt_boxes, iou_threshold=0.7)
    label_objectness = label_objectness_ignore(label_objectness, iou_above_thresh_inds)

    label_objectness = torch.from_numpy(label_objectness).to(device)
    label_location = torch.from_numpy(label_location).to(device)
    label_classification = torch.from_numpy(label_classification).to(device)

    label_objectness.detach()
    label_classification.detach()
    label_location.detach()

    scales = torch.from_numpy(scale_location).to(device)

    total_loss = get_loss(p0, label_objectness, label_location, label_classification, scales,
                          num_anchors=3, num_classes=7)
    total_loss_data = total_loss.cpu().numpy()
    print(total_loss_data)
