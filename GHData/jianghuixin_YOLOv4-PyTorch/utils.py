import os

import cv2
import torch


def plot_box(boxes, img_file, save_file):
    """
    绘制目标框
    :param boxes: torch.Tensor N*4
    :param img_file: 原始图像
    :param save_file: 基于原始图像绘制目标框
    :return: None
    """
    img = cv2.imread(img_file)

    width = img.shape[1]
    height = img.shape[0]

    for box in boxes:
        x1, y1, x2, y2 = box

        x1 = int(width * x1.item())
        x2 = int(width * x2.item())
        y1 = int(height * y1.item())
        y2 = int(height * y2.item())

        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255))

    if not save_file:
        save_file = os.path.join(os.path.dirname(img_file), "detect.png")
    cv2.imwrite(save_file, img)


def nms(boxes, confs, nms_thresh):
    conf_sorted, indices = confs.sort(descending=True)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    area = (y2 - y1) * (x2 - x1)

    keeps = []
    while indices.numel() > 0:
        idx_self = indices[0]
        idx_other = indices[1:]
        # 保留置信度最高的目标框
        keeps.append(idx_self)

        rect_x1 = torch.max(x1[idx_self], x1[idx_other])
        rect_y1 = torch.max(y1[idx_self], y1[idx_other])
        rect_x2 = torch.min(x2[idx_self], x2[idx_other])
        rect_y2 = torch.min(y2[idx_self], y2[idx_other])

        w = torch.clamp(rect_x2 - rect_x1, min=0.0)
        h = torch.clamp(rect_y2 - rect_y1, min=0.0)

        inter = w * h
        iou = inter / (area[idx_self] + area[idx_other] - inter)

        idx_other = idx_other[iou < nms_thresh]
        indices = idx_other

    return keeps


def post_processing(boxes, confs, conf_thresh, nms_thresh):
    """
    筛选目标框
    :param boxes: (B, 3*H*W, 4)
    :param confs: (B, 3*H*W, 80)
    :param conf_thresh:
    :param nms_thresh:
    :return: (boxes, probs)
    """

    batches = confs.size(0)
    num_classes = confs.size(2)
    max_confs, max_idx = confs.max(dim=2)

    idxes = []
    rects = []
    probs = []
    for batch in range(batches):
        # 每一张图像依据置信度筛选
        conf_filter = max_confs[batch] > conf_thresh
        box_b = boxes[batch][conf_filter]
        idx_b = max_idx[batch][conf_filter]
        confs_b = max_confs[batch][conf_filter]

        for c in range(num_classes):
            # 筛选当前类别
            cls_filter = (idx_b == c)
            idx_cls = idx_b[cls_filter]
            box_cls = box_b[cls_filter]
            conf_cls = confs_b[cls_filter]
            # NMS 去掉重叠目标框
            keeps = nms(box_cls, conf_cls, nms_thresh)

            for keep in keeps:
                idxes.append(idx_cls[keep])
                rects.append(box_cls[keep])
                probs.append(conf_cls[keep])

    idxes_pred = torch.stack(idxes, dim=0)
    rects_pred = torch.stack(rects, dim=0)
    probs_pred = torch.stack(probs, dim=0)

    return idxes_pred, rects_pred, probs_pred
