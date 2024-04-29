import torch
from config import config


def point_form(boxes):
    """将 cx, cy, w, h 形式的点转换为 x1, y1, x2, y2 形式
    
    Args:
        boxes: shape [N, 4]
    Return:
        shape [N, 4]
    """
    return torch.cat((boxes[:, 0:2]-boxes[:, 2:4]/2, boxes[:, 0:2]+boxes[:, 2:4]/2), dim=1)

def center_form(boxes):
    """将 x1, y1, x2, y2 形式的点转换为 cx, cy, w, h 形式
    
    Args:
        boxes: shape [N, 4]
    Return:
        shape [N, 4]
    """
    return torch.cat(((boxes[:, 0:2]+boxes[:, 2:4])/2, boxes[:, 2:4]-boxes[:, 0:2]), dim=1)


def encode(boxes, priors, config=config):
    """boxes 和 priors 为 cx, cy, w, h 形式
        boxes 与相匹配的 priors 进行 encode
    Args:
        boxes: shape [N, 4]
        priors: shape [N, 4]
    Return:
        shape [N, 4]
    """
    cxcy_target = (boxes[:, 0:2] - priors[:, 0:2]) / (config["variance"][0] * priors[:, 2:4])
    wh_target = torch.log(boxes[:, 2:4] / priors[:, 2:4]) / config["variance"][1]
    return torch.cat((cxcy_target, wh_target), dim=1)

def decode(out_loc, priors, config=config):
    """priors 为 cx, cy, w, h 形式
        priors 与 输出进行 decode
    Args:
        out_loc: shape [N, 4]
        priors: shape [N, 4]
    Return:
        boxes: shape [N, 4], 为 x1, y1, x2, y2 形式
    """
    boxes_cxcy = out_loc[:, 0:2] * config["variance"][0] * priors[:, 2:4] + priors[:, 0:2]
    boxes_wh = torch.exp(out_loc[:, 2:4] * config["variance"][1]) * priors[:, 2:4]
    boxes = torch.cat((boxes_cxcy, boxes_wh), dim=1)
    
    # 前面为解码过程
    # 转换为 x1, y1, x2, y2 形式
    boxes[:, 0:2] -= boxes[:, 2:4] / 2
    boxes[:, 2:4] += boxes[:, 0:2]
    return boxes
