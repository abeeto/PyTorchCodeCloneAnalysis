import numpy as np
import torch


def ious(box, boxes, isMin = False):
    box_area = (box[3] - box[1]) * (box[4] - box[2])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
    xx1 = torch.max(box[1], boxes[:, 1])
    yy1 = torch.max(box[2], boxes[:, 2])
    xx2 = torch.min(box[3], boxes[:, 3])
    yy2 = torch.min(box[4], boxes[:, 4])

    w = torch.clamp(xx2 - xx1, min=0)
    h = torch.clamp(yy2 - yy1, min=0)

    inter = w * h

    ovr2 = inter/ (box_area + area - inter)

    return ovr2

def nms(boxes, thresh=0.3, isMin = True):

    if boxes.shape[0] == 0:
        return np.array([])

    _boxes = boxes[(-boxes[:, 0]).argsort()]
    r_boxes = []

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]
        b_boxes = _boxes[1:]
        r_boxes.append(a_box)

        index = np.where(ious(a_box, b_boxes,isMin) < thresh)
        _boxes = b_boxes[index]
    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    return torch.stack(r_boxes)

if __name__ == '__main__':
    # a = np.array([1,1,11,11])
    # bs = np.array([[1,1,10,10],[11,11,20,20]])
    # print(iou(a,bs))

    bs = torch.tensor([[1, 1, 10, 10, 40,8], [1, 1, 9, 9, 10,9], [9, 8, 13, 20, 15,3], [6, 11, 18, 17, 13,2]])
    # print(bs[:,3].argsort())
    print(nms(bs))
