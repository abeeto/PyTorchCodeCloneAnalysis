from yolov3 import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1. / (1.0 + np.exp(-x))


# 将网络特征图输出的[tx, ty, th, tw]转化成预测框的坐标[x1, y1, x2, y2]
def get_yolo_box_xxyy(pred, anchors, num_classes, downsample):
    """
    pred是网络输出特征图转化成的numpy.ndarray
    anchors 是一个list。表示锚框的大小，
                例如 anchors = [116, 90, 156, 198, 373, 326]，表示有三个锚框，
                第一个锚框大小[w, h]是[116, 90]，第二个锚框大小是[156, 198]，第三个锚框大小是[373, 326]
    """
    batchsize = pred.shape[0]
    num_rows = pred.shape[-2]
    num_cols = pred.shape[-1]

    input_h = num_rows * downsample
    input_w = num_cols * downsample

    num_anchors = len(anchors) // 2

    # pred的形状是[N, C, H, W]，其中C = NUM_ANCHORS * (5 + NUM_CLASSES)
    # 对pred进行reshape
    pred = pred.reshape([-1, num_anchors, 5 + num_classes, num_rows, num_cols])
    pred_location = pred[:, :, 0:4, :, :]
    pred_location = np.transpose(pred_location, (0, 3, 4, 1, 2))
    anchors_this = []
    for ind in range(num_anchors):
        anchors_this.append([anchors[ind * 2], anchors[ind * 2 + 1]])
    anchors_this = np.array(anchors_this).astype('float32')

    # 最终输出数据保存在pred_box中，其形状是[N, H, W, NUM_ANCHORS, 4]，
    # 其中最后一个维度4代表位置的4个坐标
    pred_box = np.zeros(pred_location.shape)
    for n in range(batchsize):
        for i in range(num_rows):
            for j in range(num_cols):
                for k in range(num_anchors):
                    pred_box[n, i, j, k, 0] = j
                    pred_box[n, i, j, k, 1] = i
                    pred_box[n, i, j, k, 2] = anchors_this[k][0]
                    pred_box[n, i, j, k, 3] = anchors_this[k][1]

    # 这里使用相对坐标，pred_box的输出元素数值在0.~1.0之间
    pred_box[:, :, :, :, 0] = (sigmoid(pred_location[:, :, :, :, 0]) + pred_box[:, :, :, :, 0]) / num_cols
    pred_box[:, :, :, :, 1] = (sigmoid(pred_location[:, :, :, :, 1]) + pred_box[:, :, :, :, 1]) / num_rows
    pred_box[:, :, :, :, 2] = np.exp(pred_location[:, :, :, :, 2]) * pred_box[:, :, :, :, 2] / input_w
    pred_box[:, :, :, :, 3] = np.exp(pred_location[:, :, :, :, 3]) * pred_box[:, :, :, :, 3] / input_h

    # 将坐标从xywh转化成xyxy
    pred_box[:, :, :, :, 0] = pred_box[:, :, :, :, 0] - pred_box[:, :, :, :, 2] / 2.
    pred_box[:, :, :, :, 1] = pred_box[:, :, :, :, 1] - pred_box[:, :, :, :, 3] / 2.
    pred_box[:, :, :, :, 2] = pred_box[:, :, :, :, 0] + pred_box[:, :, :, :, 2]
    pred_box[:, :, :, :, 3] = pred_box[:, :, :, :, 1] + pred_box[:, :, :, :, 3]

    pred_box = np.clip(pred_box, 0., 1.0)

    return pred_box


# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# NUM_ANCHORS = 3
# NUM_CLASS = 7
# num_filters = NUM_ANCHORS * (NUM_CLASS + 5)
# with torch.no_grad():
#     backbone = DarkNet53_conv_body().to(device)
#     detection = YoloDetectionBlock(ch_in=1024, ch_out=512).to(device)
#     conv2d_pred = torch.nn.Conv2d(1024, num_filters, 1, padding=0).to(device)
#     x = torch.randn(1, 3, 640, 640).to(device)
#     x = x.float()
#     c0, c1, c2 = backbone(x)
#         # print(c0.shape)
#     route, tip = detection(c0)
#     p0 = conv2d_pred(tip)
#     reshape_p0 = torch.reshape(p0, [-1, NUM_ANCHORS, NUM_CLASS + 5, p0.shape[2], p0.shape[3]])
#     pred_objectness = reshape_p0[:, :, 4, :, :]
#     pred_objectness_probability = torch.sigmoid(pred_objectness)
#     print(pred_objectness.shape, pred_objectness_probability.shape)
# #
#     pred_location = reshape_p0[:, :, 0:4, :, :]
#     print(pred_location.shape)
#     anchors = [116, 90, 156, 198, 373, 326]
#     pred_boxes = get_yolo_box_xxyy(p0.cpu().numpy(), anchors, num_classes=7, downsample=32)
#     print(pred_boxes.shape)
#
#     pred_classification=reshape_p0[:,:,5:5+NUM_CLASS,:,:]
#     pred_classification_probility=torch.sigmoid(pred_classification)
#     print(pred_classification.shape)
