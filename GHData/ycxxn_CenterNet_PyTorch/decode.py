from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
from utils import _gather_feat, _transpose_and_gather_feat

def _topk(scores, K=40, trt=False):
    batch, cat, height, width = scores.size()
    
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds // width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).int()

    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind, trt=trt).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind, trt=trt).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind, trt=trt).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def ctdet_decode(heat, wh, reg=None, K=100, trt=False):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K, trt=trt)
    # print(scores.shape, inds.shape, clses.shape, ys.shape, xs.shape)
    if reg is not None:
        # print("1",reg.shape)
        reg = _transpose_and_gather_feat(reg, inds, trt=trt)
        # print("123",reg.shape)
        reg = reg.view(batch, K, 2)
        # print("xs:{}, ys:{}".format(xs.shape,ys.shape))
        # print(xs.view(batch, K, 1).shape, ys.view(batch, K, 1).shape)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds, trt=trt)
    
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    # print(clses.shape, scores.shape, bboxes.shape)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    # print(detections.shape)     
    return detections

if __name__ == "__main__":
    x = torch.randn(1,3,10,10).sigmoid_()
    # y = _nms(x)
    # print(x)
    # print("-----------")
    # print(y)
    # print("-----------")
    # print(x.shape, y.shape)
    # print("-----------")
    # print(_topk(y))
    # print(_topk(y)[0].shape)
    print(x)
    batch, cat, height, width = x.size()
    print("h: {},w: {}".format(height, width))
    print(x.view(batch, cat, -1).shape)
    c = torch.topk(x.view(batch, cat, -1), 40)
    # print(c.shape)
    print(c)
    print(c[0].shape,c[1].shape)
    topk_scores, topk_inds = c[0],c[1]
    print("==============")
    print(topk_inds)
    print("==============")
    topk_inds = topk_inds % (height * width)
    print(topk_inds)
    topk_ys   = (topk_inds // width).int().float()
    topk_xs   = (topk_inds % width).int().float()

    print(topk_inds)
    print(topk_inds // width)
    print(topk_inds % width)

    print("==============")
    print(topk_scores.shape,topk_scores.view(batch, -1).shape)
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), 40)
    print(topk_score.shape,topk_ind.shape)
    print(topk_score,"\n",topk_ind)
    topk_clses = (topk_ind // 40).int()
    print(topk_clses)
    print(topk_inds.shape, topk_inds.view(batch,-1,1).shape)
    print(topk_ind.shape)