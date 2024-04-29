import math
import torch
import numpy as np
from itertools import product


def get_prior(feature_map_size, image_size, min_size, max_size, ratio):
    """生成 prior\n
    prior 为 cx, cy, w, h 形式
    Args:
        feature_map_size: feature_map 的 h w
        image_size: 原始图片大小，个人觉得这就是一个 scale，不一定要是图片大小
        min_size: 小 scale
        max_size: 大 scale
        ratio: 宽高比 w / h
    Return:
        prior_boxes: 
    """
    h, w = feature_map_size

    half_grid_w, half_grid_h = 0.5 / w, 0.5 / h
    prior_boxes = []
    for y, x in product(range(h), range(w)):
        cx = x / w + half_grid_w
        cy = y / h + half_grid_h

        # 短边
        s_k = min_size / image_size
        prior_boxes += [cx, cy, s_k, s_k]
        
        # 长边
        s_k_ = math.sqrt(s_k * max_size / image_size)
        prior_boxes += [cx, cy, s_k_, s_k_]
        for r in ratio:
            prior_boxes += [cx, cy, s_k*math.sqrt(r), s_k/math.sqrt(r)]
            # prior_boxes += [cx, cy, s_k/math.sqrt(r), s_k*math.sqrt(r)]
    prior_boxes = torch.Tensor(prior_boxes).view(-1, 4)
    return prior_boxes.clamp(max=1, min=0)



if __name__ == "__main__":
    feature_map_size = [3, 3]
    image_size = 300
    min_size = 213
    max_size = 264
    ratio = [2, 1/2]
    prior = get_prior(feature_map_size, image_size, min_size, max_size, ratio)
    print(prior.shape)
    
    from config import config
    
    feature_map_sizes = [torch.Size([38, 38]), torch.Size([19, 19]), torch.Size([10, 10]), torch.Size([5, 5]), torch.Size([3, 3]), torch.Size([1, 1])]
    priors = [get_prior(feature_map_size, image_size, min_size, max_size, ratio) for feature_map_size, min_size, max_size, ratio in zip(feature_map_sizes, config["min_sizes"], config["max_sizes"], config["ratios"])]
    priors = torch.cat(priors, dim=0)
    print(priors.shape)
